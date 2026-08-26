"""Source-only trainer adapters for the v0.3 CORRO-style controls.

The public objects in this module are NumPy-only.  JAX, Flax and Optax are
imported only when :class:`JaxCorroTrainingBackend` is actually asked to fit a
model.  Merely importing/configuring the v0.3 runner therefore does not turn
the optional research stack into a completion dependency.

``R5`` delegates to the archived two-hidden-layer CORRO-style trainer.
``R5L`` uses the same episode-aware task-SupCon sampling and optimization
semantics, but replaces the MLP with one bias-free trainable linear projection
followed by L2 normalization.  Both paths accept only the exact registered
source-training rows.  Validation is also explicitly source-side; query banks
have no argument or callback through which they can enter fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from ..hashing import sha256_bytes, sha256_json, sha256_ndarrays
from .representation_ladder import (
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationLadderError,
    TrainedCallableArtifact,
    TrainingRequest,
)


CORRO_TRAINER_SCHEMA = "policy-learnware.v03-corro-trainer-adapter.v0"
CORRO_SOURCE_TASK_SCHEMA = "policy-learnware.v03-corro-source-task.v0"
CORRO_SOURCE_SPLIT_SCHEMA = "policy-learnware.v03-corro-source-split.v0"
CORRO_OPTIMIZATION_SCHEMA = "policy-learnware.v03-corro-optimization.v0"
FORMAL_CORRO_TRAINER_CONTRACT_SCHEMA = (
    "policy-learnware.v03-formal-corro-trainer-contract.v0"
)

SOURCE_REPRESENTATION_TRAIN = "source_representation_train"
SOURCE_REPRESENTATION_VALIDATION = "source_representation_validation"
SourceSplitRole = Literal[
    "source_representation_train", "source_representation_validation"
]

TASK_SUPCON_OBJECTIVE = {
    "schema": "policy-learnware.v03-task-supcon-objective.v0",
    "labels": "source_task_categorical",
    "positive_pair": "same_task_and_different_episode",
    "negative_pair": "all_nonself_pairs_not_positive",
    "similarity": "normalized_dot_product",
    "loss": "supervised_contrastive_log_softmax",
    "query_access": "forbidden",
}
TASK_SUPCON_OBJECTIVE_DIGEST = sha256_json(TASK_SUPCON_OBJECTIVE)


def jax_corro_backend_implementation_digest(representation_id: str) -> str:
    """Frozen implementation identity of the only formal v0.3 backend."""

    if representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
        payload = {
            "schema": "policy-learnware.v03-jax-corro-backend.v0",
            "architecture": "two-hidden-relu-dense-l2",
            "training_entrypoint": (
                "policy_learnware_v0.representation.encoder."
                "train_transition_encoder"
            ),
            "objective_digest": TASK_SUPCON_OBJECTIVE_DIGEST,
        }
    elif representation_id == R5L_SUPERVISED_LINEAR:
        payload = {
            "schema": "policy-learnware.v03-jax-corro-backend.v0",
            "architecture": "single-bias-free-linear-l2",
            "sampler": "episode-aware-task-balanced",
            "objective_digest": TASK_SUPCON_OBJECTIVE_DIGEST,
        }
    else:
        raise CorroTrainerError("formal CORRO backend supports only R5/R5L")
    return sha256_json(payload)


def _bound_trainer_implementation_digest(
    representation_id: str, backend_implementation_digest: str
) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v03-corro-implementation-binding.v0",
            "adapter": "source-only-corro-trainer/v0",
            "backend_implementation_digest": backend_implementation_digest,
            "representation_id": representation_id,
        }
    )


class CorroTrainerError(RepresentationLadderError):
    """The source split, training request, or backend result is invalid."""


class CorroTrainerDependencyError(ImportError):
    """Raised only when a real fit needs the optional JAX research stack."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CorroTrainerError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CorroTrainerError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _positive_int(value: Any, where: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CorroTrainerError(f"{where} must be a {qualifier} integer")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CorroTrainerError(f"{where} must be a {qualifier} integer")
    return result


def _positive_float(value: Any, where: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CorroTrainerError(f"{where} must be a finite {qualifier} number")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CorroTrainerError(f"{where} must be a finite {qualifier} number")
    return result


def _matrix(value: Any, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] <= 0:
        raise CorroTrainerError(f"{where} must be a non-empty 2D matrix")
    if raw.dtype.hasobject or not np.issubdtype(raw.dtype, np.number):
        raise CorroTrainerError(f"{where} must be numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise CorroTrainerError(f"{where} must be finite")
    result.setflags(write=False)
    return result


def _offsets(value: Any, rows: int, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise CorroTrainerError(f"{where} must be a one-dimensional integer array")
    result = np.ascontiguousarray(raw, dtype=np.int64).copy()
    if (
        result.size < 3
        or int(result[0]) != 0
        or int(result[-1]) != rows
        or np.any(np.diff(result) <= 0)
    ):
        raise CorroTrainerError(
            f"{where} must delimit at least two non-empty episodes and cover all rows"
        )
    result.setflags(write=False)
    return result


def _safe_task_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CorroTrainerError("task_id must be a non-empty stripped string")
    return value


@dataclass(frozen=True)
class CorroTaskDataset:
    """One episode-partitioned source-side task matrix."""

    task_id: str
    packed: np.ndarray
    episode_offsets: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _safe_task_id(self.task_id))
        packed = _matrix(self.packed, f"{self.task_id}.packed")
        object.__setattr__(self, "packed", packed)
        object.__setattr__(
            self,
            "episode_offsets",
            _offsets(
                self.episode_offsets,
                packed.shape[0],
                f"{self.task_id}.episode_offsets",
            ),
        )

    @property
    def input_dim(self) -> int:
        return int(self.packed.shape[1])

    @property
    def dataset_digest(self) -> str:
        return sha256_json(
            {
                "schema": CORRO_SOURCE_TASK_SCHEMA,
                "task_id": self.task_id,
                "arrays_digest": sha256_ndarrays(
                    {
                        "packed": self.packed,
                        "episode_offsets": self.episode_offsets,
                    }
                ),
            }
        )


@dataclass(frozen=True)
class CorroSourceSplit:
    """A role-labelled source train or validation split.

    Tasks are canonicalized by ``task_id`` so concatenated rows and categorical
    labels are deterministic regardless of caller mapping order.
    """

    role: SourceSplitRole
    tasks: tuple[CorroTaskDataset, ...]

    def __post_init__(self) -> None:
        if self.role not in {
            SOURCE_REPRESENTATION_TRAIN,
            SOURCE_REPRESENTATION_VALIDATION,
        }:
            raise CorroTrainerError("unsupported CORRO source split role")
        raw_tasks = tuple(self.tasks)
        if len(raw_tasks) < 2 or not all(
            isinstance(item, CorroTaskDataset) for item in raw_tasks
        ):
            raise CorroTrainerError(
                "CORRO source split requires at least two typed task datasets"
            )
        tasks = tuple(sorted(raw_tasks, key=lambda item: item.task_id))
        task_ids = tuple(item.task_id for item in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise CorroTrainerError("CORRO source split task IDs must be unique")
        dimensions = {item.input_dim for item in tasks}
        if len(dimensions) != 1:
            raise CorroTrainerError(
                "all task matrices must already share one canonical input width"
            )
        object.__setattr__(self, "tasks", tasks)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(item.task_id for item in self.tasks)

    @property
    def input_dim(self) -> int:
        return self.tasks[0].input_dim

    @property
    def row_count(self) -> int:
        return sum(int(item.packed.shape[0]) for item in self.tasks)

    @property
    def split_digest(self) -> str:
        return sha256_json(
            {
                "schema": CORRO_SOURCE_SPLIT_SCHEMA,
                "role": self.role,
                "input_dim": self.input_dim,
                "tasks": [
                    {
                        "task_id": item.task_id,
                        "dataset_digest": item.dataset_digest,
                    }
                    for item in self.tasks
                ],
            }
        )

    def task_mapping(self) -> Mapping[str, CorroTaskDataset]:
        return MappingProxyType({item.task_id: item for item in self.tasks})

    def flattened_values(self) -> np.ndarray:
        result = np.ascontiguousarray(
            np.concatenate([item.packed for item in self.tasks], axis=0),
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    def flattened_task_names(self) -> np.ndarray:
        width = max(len(item.task_id) for item in self.tasks)
        result = np.concatenate(
            [
                np.full(item.packed.shape[0], item.task_id, dtype=f"<U{width}")
                for item in self.tasks
            ]
        )
        result.setflags(write=False)
        return result

    def flattened_task_indices(self) -> np.ndarray:
        result = np.concatenate(
            [
                np.full(item.packed.shape[0], index, dtype=np.int64)
                for index, item in enumerate(self.tasks)
            ]
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class CorroOptimizationConfig:
    """Shared optimization settings for R5 and its matched R5L control."""

    temperature: float = 0.1
    batch_size: int = 1024
    train_steps: int = 20_000
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    validation_interval: int = 500
    validation_batches: int = 8
    inference_batch_size: int = 8192

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "temperature", _positive_float(self.temperature, "temperature")
        )
        object.__setattr__(
            self, "batch_size", _positive_int(self.batch_size, "batch_size")
        )
        object.__setattr__(
            self,
            "train_steps",
            _positive_int(self.train_steps, "train_steps", allow_zero=True),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_float(self.learning_rate, "learning_rate"),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _positive_float(self.weight_decay, "weight_decay", allow_zero=True),
        )
        object.__setattr__(
            self,
            "validation_interval",
            _positive_int(self.validation_interval, "validation_interval"),
        )
        object.__setattr__(
            self,
            "validation_batches",
            _positive_int(self.validation_batches, "validation_batches"),
        )
        object.__setattr__(
            self,
            "inference_batch_size",
            _positive_int(self.inference_batch_size, "inference_batch_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CORRO_OPTIMIZATION_SCHEMA,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "train_steps": self.train_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
            "inference_batch_size": self.inference_batch_size,
        }

    @property
    def optimization_digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class CorroBackendResult:
    checkpoint_bytes: bytes
    implementation_digest: str
    transform: Callable[[np.ndarray], np.ndarray] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_bytes, bytes) or not self.checkpoint_bytes:
            raise CorroTrainerError("CORRO backend checkpoint_bytes cannot be empty")
        if (
            not isinstance(self.implementation_digest, str)
            or len(self.implementation_digest) != 64
            or self.implementation_digest.lower() != self.implementation_digest
        ):
            raise CorroTrainerError(
                "CORRO backend implementation_digest must be a SHA-256 digest"
            )
        try:
            int(self.implementation_digest, 16)
        except ValueError as error:
            raise CorroTrainerError(
                "CORRO backend implementation_digest must be a SHA-256 digest"
            ) from error
        if not callable(self.transform):
            raise CorroTrainerError("CORRO backend transform must be callable")


@dataclass(frozen=True)
class FormalCorroTrainerContract:
    """Persistable proof coordinate for a production R5/R5L trainer path.

    The contract is obtainable from :meth:`CorroTrainerAdapter.formal_contract`
    only when the adapter uses its built-in JAX backend (no injected backend),
    the exact formal source splits, and the frozen optimization schedule.
    """

    representation_id: str
    training_request_digest: str
    source_fit_batch_digest: str
    train_split_digest: str
    validation_split_digest: str
    optimization_digest: str
    objective_digest: str
    adapter_digest: str
    backend_implementation_digest: str
    trainer_implementation_digest: str
    contract_digest: str | None = None
    schema: str = FORMAL_CORRO_TRAINER_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_CORRO_TRAINER_CONTRACT_SCHEMA:
            raise CorroTrainerError("unsupported formal CORRO trainer contract")
        if self.representation_id not in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            raise CorroTrainerError("formal trainer contract supports only R5/R5L")
        for name in (
            "training_request_digest",
            "source_fit_batch_digest",
            "train_split_digest",
            "validation_split_digest",
            "optimization_digest",
            "objective_digest",
            "adapter_digest",
            "backend_implementation_digest",
            "trainer_implementation_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.objective_digest != TASK_SUPCON_OBJECTIVE_DIGEST:
            raise CorroTrainerError("formal trainer contract objective drifted")
        expected_backend = jax_corro_backend_implementation_digest(
            self.representation_id
        )
        if self.backend_implementation_digest != expected_backend:
            raise CorroTrainerError("formal trainer backend implementation drifted")
        if self.trainer_implementation_digest != _bound_trainer_implementation_digest(
            self.representation_id, expected_backend
        ):
            raise CorroTrainerError("formal trainer implementation binding drifted")
        expected = sha256_json(self._payload_without_digest())
        if self.contract_digest is None:
            object.__setattr__(self, "contract_digest", expected)
        elif _digest(self.contract_digest, "contract_digest") != expected:
            raise CorroTrainerError("formal trainer contract digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_id": self.representation_id,
            "training_request_digest": self.training_request_digest,
            "source_fit_batch_digest": self.source_fit_batch_digest,
            "train_split_digest": self.train_split_digest,
            "validation_split_digest": self.validation_split_digest,
            "optimization_digest": self.optimization_digest,
            "objective_digest": self.objective_digest,
            "adapter_digest": self.adapter_digest,
            "backend": "JaxCorroTrainingBackend(default)",
            "backend_implementation_digest": self.backend_implementation_digest,
            "trainer_implementation_digest": self.trainer_implementation_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "contract_digest": self.contract_digest}

    def expected_artifact_parameter_digest(self, checkpoint_bytes: bytes) -> str:
        if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
            raise CorroTrainerError("formal checkpoint bytes cannot be empty")
        return sha256_json(
            {
                "schema": "policy-learnware.v03-corro-parameter-binding.v0",
                "checkpoint_digest": sha256_bytes(checkpoint_bytes),
                "request_digest": self.training_request_digest,
                "train_split_digest": self.train_split_digest,
                "validation_split_digest": self.validation_split_digest,
                "optimization_digest": self.optimization_digest,
            }
        )

    def expected_manifest_parameter_digest(self, checkpoint_bytes: bytes) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-trained-parameter-binding.v0",
                "trainer_parameter_digest": self.expected_artifact_parameter_digest(
                    checkpoint_bytes
                ),
                "training_request_digest": self.training_request_digest,
            }
        )


@runtime_checkable
class CorroTrainingBackend(Protocol):
    def train(
        self,
        *,
        train: Mapping[str, CorroTaskDataset],
        validation: Mapping[str, CorroTaskDataset],
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult: ...

    def restore(
        self,
        *,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult: ...


@dataclass(frozen=True)
class CorroTrainerAdapter:
    """Adapt registered source splits to ``RepresentationTrainer``."""

    train_split: CorroSourceSplit
    validation_split: CorroSourceSplit
    optimization: CorroOptimizationConfig = field(
        default_factory=CorroOptimizationConfig
    )
    backend: CorroTrainingBackend | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.train_split, CorroSourceSplit) or not isinstance(
            self.validation_split, CorroSourceSplit
        ):
            raise CorroTrainerError("trainer adapter requires typed CORRO splits")
        if self.train_split.role != SOURCE_REPRESENTATION_TRAIN:
            raise CorroTrainerError("train_split has the wrong source role")
        if self.validation_split.role != SOURCE_REPRESENTATION_VALIDATION:
            raise CorroTrainerError("validation_split has the wrong source role")
        if self.train_split.task_ids != self.validation_split.task_ids:
            raise CorroTrainerError("source train/validation task sets must match")
        if self.train_split.input_dim != self.validation_split.input_dim:
            raise CorroTrainerError("source train/validation input widths must match")
        for task_id in self.train_split.task_ids:
            train_task = self.train_split.task_mapping()[task_id]
            validation_task = self.validation_split.task_mapping()[task_id]
            if train_task.dataset_digest == validation_task.dataset_digest:
                raise CorroTrainerError(
                    f"{task_id}: source train and validation datasets must be distinct"
                )
        if not isinstance(self.optimization, CorroOptimizationConfig):
            raise CorroTrainerError("optimization must be CorroOptimizationConfig")
        if self.optimization.batch_size // len(self.train_split.tasks) < 2:
            raise CorroTrainerError(
                "batch_size must provide at least two samples per source task"
            )
        if self.backend is not None and not isinstance(
            self.backend, CorroTrainingBackend
        ):
            raise CorroTrainerError("backend does not implement CorroTrainingBackend")

    @property
    def adapter_digest(self) -> str:
        return sha256_json(
            {
                "schema": CORRO_TRAINER_SCHEMA,
                "train_split_digest": self.train_split.split_digest,
                "validation_split_digest": self.validation_split.split_digest,
                "optimization_digest": self.optimization.optimization_digest,
                "objective_digest": TASK_SUPCON_OBJECTIVE_DIGEST,
            }
        )

    def formal_contract(
        self,
        *,
        request: TrainingRequest,
        source_fit_batch_digest: str,
        expected_train_split: CorroSourceSplit,
        expected_validation_split: CorroSourceSplit,
        expected_optimization: CorroOptimizationConfig,
    ) -> FormalCorroTrainerContract:
        """Bind the adapter to the production-only formal training route.

        Injected backends are intentionally a development-smoke capability.
        Formal publication uses the adapter's implicit, exact
        :class:`JaxCorroTrainingBackend`, so a callable that merely returns a
        syntactically valid ``TrainedCallableArtifact`` cannot be published as
        a formal R5/R5L fit.
        """

        if self.backend is not None:
            raise CorroTrainerError(
                "formal CORRO training forbids injected development backends"
            )
        if not isinstance(request, TrainingRequest):
            raise CorroTrainerError("formal trainer requires TrainingRequest")
        if request.objective_digest != TASK_SUPCON_OBJECTIVE_DIGEST:
            raise CorroTrainerError("formal trainer objective differs from task-SupCon")
        if not isinstance(expected_train_split, CorroSourceSplit) or not isinstance(
            expected_validation_split, CorroSourceSplit
        ):
            raise CorroTrainerError("formal trainer requires typed source splits")
        if (
            self.train_split.split_digest != expected_train_split.split_digest
            or self.validation_split.split_digest
            != expected_validation_split.split_digest
        ):
            raise CorroTrainerError(
                "formal trainer source rows differ from source-fit authority"
            )
        if not isinstance(expected_optimization, CorroOptimizationConfig):
            raise CorroTrainerError("formal trainer optimization must be typed")
        if (
            self.optimization.optimization_digest
            != expected_optimization.optimization_digest
        ):
            raise CorroTrainerError(
                "formal trainer optimization differs from representation freeze"
            )
        if (
            request.input_dim != self.train_split.input_dim
            or request.representation_id
            not in {R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR}
        ):
            raise CorroTrainerError("formal trainer request differs from source split")
        backend_digest = jax_corro_backend_implementation_digest(
            request.representation_id
        )
        return FormalCorroTrainerContract(
            representation_id=request.representation_id,
            training_request_digest=request.request_digest,
            source_fit_batch_digest=_digest(
                source_fit_batch_digest, "source_fit_batch_digest"
            ),
            train_split_digest=self.train_split.split_digest,
            validation_split_digest=self.validation_split.split_digest,
            optimization_digest=self.optimization.optimization_digest,
            objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
            adapter_digest=self.adapter_digest,
            backend_implementation_digest=backend_digest,
            trainer_implementation_digest=_bound_trainer_implementation_digest(
                request.representation_id, backend_digest
            ),
        )

    def _validate_source_call(
        self,
        source_values: np.ndarray,
        labels: np.ndarray,
        request: TrainingRequest,
    ) -> None:
        if not isinstance(request, TrainingRequest):
            raise CorroTrainerError("request must be a TrainingRequest")
        if request.objective_digest != TASK_SUPCON_OBJECTIVE_DIGEST:
            raise CorroTrainerError(
                "R5/R5L require the frozen episode-aware task-SupCon objective"
            )
        if request.input_dim != self.train_split.input_dim:
            raise CorroTrainerError("training request input width differs from source")
        candidate = _matrix(source_values, "source_values")
        expected = self.train_split.flattened_values()
        if candidate.shape != expected.shape or not np.array_equal(candidate, expected):
            raise CorroTrainerError(
                "trainer accepts only the exact registered source-training rows; "
                "query/development rows are forbidden during fit"
            )
        raw_labels = np.asarray(labels)
        if raw_labels.ndim != 1 or raw_labels.shape != (expected.shape[0],):
            raise CorroTrainerError("source task labels have the wrong shape")
        names_match = raw_labels.dtype.kind in "US" and np.array_equal(
            raw_labels.astype(str), self.train_split.flattened_task_names()
        )
        indices_match = raw_labels.dtype.kind in "iu" and np.array_equal(
            raw_labels.astype(np.int64), self.train_split.flattened_task_indices()
        )
        if not (names_match or indices_match):
            raise CorroTrainerError(
                "source task labels must match the registered categorical task blocks"
            )

    def __call__(
        self,
        source_values: np.ndarray,
        labels: np.ndarray,
        request: TrainingRequest,
    ) -> TrainedCallableArtifact:
        self._validate_source_call(source_values, labels, request)
        backend: CorroTrainingBackend = self.backend or JaxCorroTrainingBackend()
        result = backend.train(
            train=self.train_split.task_mapping(),
            validation=self.validation_split.task_mapping(),
            request=request,
            optimization=self.optimization,
        )
        return self._bind_backend_result(result, request)

    def _bind_backend_result(
        self,
        result: CorroBackendResult,
        request: TrainingRequest,
    ) -> TrainedCallableArtifact:
        if not isinstance(result, CorroBackendResult):
            raise CorroTrainerError("CORRO backend returned an invalid result")
        checkpoint_digest = sha256_bytes(result.checkpoint_bytes)
        parameter_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-corro-parameter-binding.v0",
                "checkpoint_digest": checkpoint_digest,
                "request_digest": request.request_digest,
                "train_split_digest": self.train_split.split_digest,
                "validation_split_digest": self.validation_split.split_digest,
                "optimization_digest": self.optimization.optimization_digest,
            }
        )
        implementation_digest = _bound_trainer_implementation_digest(
            request.representation_id, result.implementation_digest
        )
        return TrainedCallableArtifact(
            checkpoint_bytes=result.checkpoint_bytes,
            parameter_digest=parameter_digest,
            trainer_implementation_digest=implementation_digest,
            transform=result.transform,
        )

    def restore(
        self,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
    ) -> TrainedCallableArtifact:
        """Restore a frozen R5/R5L transform without running optimization."""

        if not isinstance(request, TrainingRequest):
            raise CorroTrainerError("restore request must be a TrainingRequest")
        if request.objective_digest != TASK_SUPCON_OBJECTIVE_DIGEST:
            raise CorroTrainerError(
                "R5/R5L restore requires the frozen task-SupCon objective"
            )
        if request.input_dim != self.train_split.input_dim:
            raise CorroTrainerError("restore input width differs from registered source")
        if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
            raise CorroTrainerError("restore checkpoint_bytes must be non-empty")
        backend: CorroTrainingBackend = self.backend or JaxCorroTrainingBackend()
        result = backend.restore(
            checkpoint_bytes=checkpoint_bytes,
            request=request,
            optimization=self.optimization,
        )
        if result.checkpoint_bytes != checkpoint_bytes:
            raise CorroTrainerError("CORRO backend restore changed checkpoint bytes")
        return self._bind_backend_result(result, request)


@dataclass(frozen=True)
class JaxCorroTrainingBackend:
    """Real backend; every optional dependency is loaded inside ``train``."""

    def train(
        self,
        *,
        train: Mapping[str, CorroTaskDataset],
        validation: Mapping[str, CorroTaskDataset],
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        if request.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
            return self._train_r5(
                train=train,
                validation=validation,
                request=request,
                optimization=optimization,
            )
        if request.representation_id == R5L_SUPERVISED_LINEAR:
            return self._train_r5l(
                train=train,
                validation=validation,
                request=request,
                optimization=optimization,
            )
        raise CorroTrainerError("JAX CORRO backend received an unsupported request")

    def restore(
        self,
        *,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        if request.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
            return self._restore_r5(
                checkpoint_bytes=checkpoint_bytes,
                request=request,
                optimization=optimization,
            )
        if request.representation_id == R5L_SUPERVISED_LINEAR:
            return self._restore_r5l(
                checkpoint_bytes=checkpoint_bytes,
                request=request,
                optimization=optimization,
            )
        raise CorroTrainerError("JAX CORRO backend received an unsupported restore request")

    @staticmethod
    def _restore_r5(
        *,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        try:
            from flax import serialization

            from ..representation.encoder import (
                EncoderCheckpoint,
                EncoderConfig,
                TransitionSemanticEncoder,
            )
        except ImportError as error:
            raise CorroTrainerDependencyError(
                "R5 restore requires the optional JAX and Flax research stack"
            ) from error
        config = EncoderConfig(
            input_dim=request.input_dim,
            hidden_dims=request.hidden_dims,
            latent_dim=request.output_dim,
            activation="relu",
            l2_normalize_output=True,
            temperature=optimization.temperature,
            batch_size=optimization.batch_size,
            train_steps=optimization.train_steps,
            learning_rate=optimization.learning_rate,
            weight_decay=optimization.weight_decay,
            validation_interval=optimization.validation_interval,
            validation_batches=optimization.validation_batches,
            seed=request.seed,
        )
        template = TransitionSemanticEncoder.initialize(config).checkpoint.params
        try:
            params = serialization.from_bytes(template, checkpoint_bytes)
        except Exception as error:
            raise CorroTrainerError("R5 checkpoint bytes cannot be restored") from error
        encoder = TransitionSemanticEncoder(EncoderCheckpoint(config=config, params=params))

        def transform(values: np.ndarray) -> np.ndarray:
            return encoder.encode(values, batch_size=optimization.inference_batch_size)

        return CorroBackendResult(
            checkpoint_bytes=checkpoint_bytes,
            implementation_digest=jax_corro_backend_implementation_digest(
                R5_VIEW_SPECIFIC_CORRO_REFIT
            ),
            transform=transform,
        )

    @staticmethod
    def _restore_r5l(
        *,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            from flax import serialization
        except ImportError as error:
            raise CorroTrainerDependencyError(
                "R5L restore requires the optional JAX and Flax research stack"
            ) from error

        class LinearProjection(nn.Module):
            output_dim: int

            @nn.compact
            def __call__(self, value: Any) -> Any:
                value = nn.Dense(self.output_dim, use_bias=False)(value)
                denominator = jnp.maximum(
                    jnp.linalg.norm(value, axis=-1, keepdims=True), 1.0e-12
                )
                return value / denominator

        model = LinearProjection(output_dim=request.output_dim)
        template = model.init(
            jax.random.PRNGKey(request.seed),
            jnp.zeros((1, request.input_dim), dtype=jnp.float32),
        )["params"]
        try:
            params = serialization.from_bytes(template, checkpoint_bytes)
        except Exception as error:
            raise CorroTrainerError("R5L checkpoint bytes cannot be restored") from error

        def transform(values: np.ndarray) -> np.ndarray:
            matrix = np.asarray(values, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != request.input_dim:
                raise CorroTrainerError(
                    f"R5L transform requires shape [N,{request.input_dim}]"
                )
            chunks = []
            for start in range(0, matrix.shape[0], optimization.inference_batch_size):
                chunks.append(
                    np.asarray(
                        model.apply(
                            {"params": params},
                            jnp.asarray(
                                matrix[start : start + optimization.inference_batch_size]
                            ),
                        ),
                        dtype=np.float32,
                    )
                )
            if not chunks:
                return np.empty((0, request.output_dim), dtype=np.float32)
            return np.concatenate(chunks, axis=0)

        return CorroBackendResult(
            checkpoint_bytes=checkpoint_bytes,
            implementation_digest=jax_corro_backend_implementation_digest(
                R5L_SUPERVISED_LINEAR
            ),
            transform=transform,
        )

    @staticmethod
    def _train_r5(
        *,
        train: Mapping[str, CorroTaskDataset],
        validation: Mapping[str, CorroTaskDataset],
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        try:
            from flax import serialization

            from ..representation.encoder import (
                EncoderConfig,
                TransitionSemanticEncoder,
                train_transition_encoder,
            )
        except ImportError as error:
            raise CorroTrainerDependencyError(
                "R5 fitting requires the optional JAX, Flax and Optax research stack"
            ) from error

        config = EncoderConfig(
            input_dim=request.input_dim,
            hidden_dims=request.hidden_dims,
            latent_dim=request.output_dim,
            activation="relu",
            l2_normalize_output=True,
            temperature=optimization.temperature,
            batch_size=optimization.batch_size,
            train_steps=optimization.train_steps,
            learning_rate=optimization.learning_rate,
            weight_decay=optimization.weight_decay,
            validation_interval=optimization.validation_interval,
            validation_batches=optimization.validation_batches,
            seed=request.seed,
        )
        try:
            checkpoint = train_transition_encoder(train, validation, config)
        except ImportError as error:
            raise CorroTrainerDependencyError(
                "R5 fitting requires the optional JAX, Flax and Optax research stack"
            ) from error
        checkpoint_bytes = bytes(serialization.to_bytes(checkpoint.params))
        encoder = TransitionSemanticEncoder(checkpoint)

        def transform(values: np.ndarray) -> np.ndarray:
            return encoder.encode(
                values, batch_size=optimization.inference_batch_size
            )

        return CorroBackendResult(
            checkpoint_bytes=checkpoint_bytes,
            implementation_digest=jax_corro_backend_implementation_digest(
                R5_VIEW_SPECIFIC_CORRO_REFIT
            ),
            transform=transform,
        )

    @staticmethod
    def _train_r5l(
        *,
        train: Mapping[str, CorroTaskDataset],
        validation: Mapping[str, CorroTaskDataset],
        request: TrainingRequest,
        optimization: CorroOptimizationConfig,
    ) -> CorroBackendResult:
        try:
            import jax
            import jax.numpy as jnp
            from flax import linen as nn
            from flax import serialization
            import optax

            from ..representation.contrastive import (
                TaskBalancedBatchSampler,
                supervised_contrastive_loss_jax,
            )
        except ImportError as error:
            raise CorroTrainerDependencyError(
                "R5L fitting requires the optional JAX, Flax and Optax research stack"
            ) from error

        class LinearProjection(nn.Module):
            output_dim: int

            @nn.compact
            def __call__(self, value: Any) -> Any:
                value = nn.Dense(self.output_dim, use_bias=False)(value)
                denominator = jnp.maximum(
                    jnp.linalg.norm(value, axis=-1, keepdims=True), 1.0e-12
                )
                return value / denominator

        train_sampler = TaskBalancedBatchSampler(
            train, batch_size=optimization.batch_size, seed=request.seed
        )
        validation_sampler = TaskBalancedBatchSampler(
            validation,
            batch_size=optimization.batch_size,
            seed=request.seed + 1,
        )
        frozen_validation_batches = tuple(
            validation_sampler.sample()
            for _ in range(optimization.validation_batches)
        )
        model = LinearProjection(output_dim=request.output_dim)
        params = model.init(
            jax.random.PRNGKey(request.seed),
            jnp.zeros((1, request.input_dim), dtype=jnp.float32),
        )["params"]
        optimizer = optax.adamw(
            learning_rate=optimization.learning_rate,
            weight_decay=optimization.weight_decay,
        )
        optimizer_state = optimizer.init(params)

        @jax.jit
        def step(
            current: Any,
            state: Any,
            values: Any,
            tasks: Any,
            episodes: Any,
        ) -> tuple[Any, Any, Any]:
            def objective(candidate: Any) -> Any:
                embeddings = model.apply({"params": candidate}, values)
                return supervised_contrastive_loss_jax(
                    embeddings,
                    tasks,
                    episodes,
                    temperature=optimization.temperature,
                )

            loss, gradients = jax.value_and_grad(objective)(current)
            updates, new_state = optimizer.update(gradients, state, current)
            return optax.apply_updates(current, updates), new_state, loss

        @jax.jit
        def evaluate(
            current: Any, values: Any, tasks: Any, episodes: Any
        ) -> Any:
            embeddings = model.apply({"params": current}, values)
            return supervised_contrastive_loss_jax(
                embeddings,
                tasks,
                episodes,
                temperature=optimization.temperature,
            )

        best_params = params
        best_validation = float("inf")
        for step_index in range(1, optimization.train_steps + 1):
            batch = train_sampler.sample()
            params, optimizer_state, _ = step(
                params,
                optimizer_state,
                jnp.asarray(batch.transitions),
                jnp.asarray(batch.task_labels),
                jnp.asarray(batch.episode_ids),
            )
            if (
                step_index % optimization.validation_interval == 0
                or step_index == optimization.train_steps
            ):
                validation_loss = float(
                    np.mean(
                        [
                            float(
                                evaluate(
                                    params,
                                    jnp.asarray(batch.transitions),
                                    jnp.asarray(batch.task_labels),
                                    jnp.asarray(batch.episode_ids),
                                )
                            )
                            for batch in frozen_validation_batches
                        ]
                    )
                )
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    best_params = jax.tree_util.tree_map(
                        lambda value: value.copy(), params
                    )

        checkpoint_bytes = bytes(serialization.to_bytes(best_params))

        def transform(values: np.ndarray) -> np.ndarray:
            matrix = np.asarray(values, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != request.input_dim:
                raise CorroTrainerError(
                    f"R5L transform requires shape [N,{request.input_dim}]"
                )
            chunks = []
            for start in range(0, matrix.shape[0], optimization.inference_batch_size):
                chunks.append(
                    np.asarray(
                        model.apply(
                            {"params": best_params},
                            jnp.asarray(
                                matrix[
                                    start : start + optimization.inference_batch_size
                                ]
                            ),
                        ),
                        dtype=np.float32,
                    )
                )
            if not chunks:
                return np.empty((0, request.output_dim), dtype=np.float32)
            return np.concatenate(chunks, axis=0)

        return CorroBackendResult(
            checkpoint_bytes=checkpoint_bytes,
            implementation_digest=jax_corro_backend_implementation_digest(
                R5L_SUPERVISED_LINEAR
            ),
            transform=transform,
        )


__all__ = [
    "CORRO_OPTIMIZATION_SCHEMA",
    "CORRO_SOURCE_SPLIT_SCHEMA",
    "CORRO_SOURCE_TASK_SCHEMA",
    "CORRO_TRAINER_SCHEMA",
    "FORMAL_CORRO_TRAINER_CONTRACT_SCHEMA",
    "SOURCE_REPRESENTATION_TRAIN",
    "SOURCE_REPRESENTATION_VALIDATION",
    "TASK_SUPCON_OBJECTIVE",
    "TASK_SUPCON_OBJECTIVE_DIGEST",
    "CorroBackendResult",
    "FormalCorroTrainerContract",
    "CorroOptimizationConfig",
    "CorroSourceSplit",
    "CorroTaskDataset",
    "CorroTrainerDependencyError",
    "CorroTrainerError",
    "CorroTrainerAdapter",
    "CorroTrainingBackend",
    "JaxCorroTrainingBackend",
    "jax_corro_backend_implementation_digest",
]
