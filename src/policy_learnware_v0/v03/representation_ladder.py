"""Deterministic, source-fitted representation controls for v0.3.

This module deliberately contains no runner, artifact writer, GPU dependency,
or encoder-family migration hook.  It implements the small representation
ladder used by the v0.3 signal atlas and exposes digest-bound value objects that
can be persisted by a later formal runner.

Every fitting entry point accepts only :class:`RepresentationBatch` objects
with role ``SOURCE_FIT``.  Query batches are accepted only by the frozen
``transform`` method.  Consequently an injected trainer never receives a
query/oracle object through this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import sha256_bytes, sha256_json, sha256_ndarrays
from .signal_controls import HistoricalRandomTanhSpec


REPRESENTATION_LADDER_SCHEMA = "policy-learnware.v03-representation-manifest.v0"
REPRESENTATION_OUTPUT_SCHEMA = "policy-learnware.v03-representation-output.v0"
TRAINING_REQUEST_SCHEMA = "policy-learnware.v03-training-request.v0"
FORMAL_TRAINED_REPRESENTATION_RECEIPT_SCHEMA = (
    "policy-learnware.v03-formal-trained-representation-receipt.v0"
)

R0_PADDED_RAW = "R0_PADDED_RAW"
R1_FIXED_RANDOM_LINEAR = "R1_FIXED_RANDOM_LINEAR"
R2_SOURCE_PCA_WHITEN = "R2_SOURCE_PCA_WHITEN"
R3_MATCHED_RANDOM_MLP = "R3_MATCHED_RANDOM_MLP"
R4_ARCHIVED_FROZEN_CORRO = "R4_ARCHIVED_FROZEN_CORRO"
R5_VIEW_SPECIFIC_CORRO_REFIT = "R5_VIEW_SPECIFIC_CORRO_REFIT"
R5L_SUPERVISED_LINEAR = "R5L_SUPERVISED_LINEAR"
R_HIST_RANDOM_TANH = "R_HIST_RANDOM_TANH"

REPRESENTATION_IDS = frozenset(
    {
        R0_PADDED_RAW,
        R1_FIXED_RANDOM_LINEAR,
        R2_SOURCE_PCA_WHITEN,
        R3_MATCHED_RANDOM_MLP,
        R4_ARCHIVED_FROZEN_CORRO,
        R5_VIEW_SPECIFIC_CORRO_REFIT,
        R5L_SUPERVISED_LINEAR,
        R_HIST_RANDOM_TANH,
    }
)

ARCHITECTURES: Mapping[str, str] = MappingProxyType(
    {
        R0_PADDED_RAW: "IDENTITY",
        R1_FIXED_RANDOM_LINEAR: "LINEAR_NO_BIAS_NO_NONLINEARITY",
        R2_SOURCE_PCA_WHITEN: "SOURCE_PCA_FIXED_SIGN",
        R3_MATCHED_RANDOM_MLP: "MATCHED_TWO_HIDDEN_RELU_L2",
        R4_ARCHIVED_FROZEN_CORRO: "ARCHIVED_TWO_HIDDEN_RELU_L2",
        R5_VIEW_SPECIFIC_CORRO_REFIT: "CORRO_STYLE_TWO_HIDDEN_RELU_L2",
        R5L_SUPERVISED_LINEAR: "SUPERVISED_LINEAR_L2",
        R_HIST_RANDOM_TANH: "SINGLE_AFFINE_TANH",
    }
)

BatchRole = Literal["SOURCE_FIT", "QUERY_TRANSFORM"]
_BATCH_ROLES = frozenset({"SOURCE_FIT", "QUERY_TRANSFORM"})


class RepresentationLadderError(ValueError):
    """A representation fit, transform, or persisted manifest is invalid."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise RepresentationLadderError(
            f"{where} must be a lowercase SHA-256 digest"
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise RepresentationLadderError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RepresentationLadderError(f"{where} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise RepresentationLadderError(f"{where} must be a positive integer")
    return result


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise RepresentationLadderError("seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise RepresentationLadderError("seed must be a non-negative integer")
    return result


def _finite_matrix(value: Any, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] <= 0:
        raise RepresentationLadderError(f"{where} must be a non-empty 2D array")
    if raw.dtype.hasobject or not np.issubdtype(raw.dtype, np.number):
        raise RepresentationLadderError(f"{where} must be numeric")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RepresentationLadderError(f"{where} must be finite")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _finite_vector(value: Any, where: str, *, length: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] <= 0:
        raise RepresentationLadderError(f"{where} must be a non-empty vector")
    if length is not None and raw.shape != (length,):
        raise RepresentationLadderError(f"{where} must have shape ({length},)")
    if raw.dtype.hasobject or not np.issubdtype(raw.dtype, np.number):
        raise RepresentationLadderError(f"{where} must be numeric")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RepresentationLadderError(f"{where} must be finite")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _strict(value: Mapping[str, Any], fields: set[str], where: str) -> None:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise RepresentationLadderError(f"{where} must be a string-keyed mapping")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise RepresentationLadderError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float64).eps)


@dataclass(frozen=True)
class RepresentationBatch:
    """One immutable source-fit or query-transform feature matrix."""

    values: np.ndarray
    dataset_digest: str
    role: BatchRole

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _finite_matrix(self.values, "values"))
        object.__setattr__(
            self, "dataset_digest", _digest(self.dataset_digest, "dataset_digest")
        )
        if self.role not in _BATCH_ROLES:
            raise RepresentationLadderError(f"unsupported batch role: {self.role!r}")

    @property
    def input_dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def batch_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-representation-batch.v0",
                "dataset_digest": self.dataset_digest,
                "role": self.role,
                "arrays_digest": sha256_ndarrays({"values": self.values}),
            }
        )


@dataclass(frozen=True)
class RepresentationManifest:
    """Persistable coordinate-system identity for one frozen transform."""

    representation_id: str
    architecture: str
    input_dim: int
    output_dim: int
    protocol_digest: str
    params_digest: str
    source_fit_digest: str
    implementation_digest: str
    checkpoint_digest: str | None
    seed: int | None
    coordinate_digest: str | None = None
    schema: str = REPRESENTATION_LADDER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTATION_LADDER_SCHEMA:
            raise RepresentationLadderError("unsupported representation manifest schema")
        if self.representation_id not in REPRESENTATION_IDS:
            raise RepresentationLadderError("unknown representation_id")
        if self.architecture != ARCHITECTURES[self.representation_id]:
            raise RepresentationLadderError(
                "representation architecture disagrees with the frozen ID"
            )
        object.__setattr__(self, "input_dim", _positive_int(self.input_dim, "input_dim"))
        object.__setattr__(
            self, "output_dim", _positive_int(self.output_dim, "output_dim")
        )
        for name in (
            "protocol_digest",
            "params_digest",
            "source_fit_digest",
            "implementation_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.checkpoint_digest is not None:
            object.__setattr__(
                self,
                "checkpoint_digest",
                _digest(self.checkpoint_digest, "checkpoint_digest"),
            )
        if self.seed is not None:
            object.__setattr__(self, "seed", _seed(self.seed))

        requires_checkpoint = self.representation_id != R0_PADDED_RAW
        if requires_checkpoint != (self.checkpoint_digest is not None):
            raise RepresentationLadderError(
                "all non-identity representations require a checkpoint binding"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.coordinate_digest is None:
            object.__setattr__(self, "coordinate_digest", expected)
        elif _digest(self.coordinate_digest, "coordinate_digest") != expected:
            raise RepresentationLadderError(
                "coordinate_digest does not match representation manifest"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_id": self.representation_id,
            "architecture": self.architecture,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "protocol_digest": self.protocol_digest,
            "params_digest": self.params_digest,
            "source_fit_digest": self.source_fit_digest,
            "implementation_digest": self.implementation_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "coordinate_digest": self.coordinate_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RepresentationManifest":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "representation manifest")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class FormalTrainedRepresentationReceipt:
    """Portable join from a formal R5/R5L checkpoint to represented banks."""

    representation_id: str
    representation_coordinate_digest: str
    checkpoint_artifact_digest: str
    checkpoint_manifest_digest: str
    training_request_digest: str
    representation_execution_plan_digest: str
    formal_source_fit_batch_digest: str
    formal_trainer_contract_digest: str
    formal_fit_job_digest: str
    formal_source_fit_schedule_digest: str
    receipt_digest: str | None = None
    schema: str = FORMAL_TRAINED_REPRESENTATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_TRAINED_REPRESENTATION_RECEIPT_SCHEMA:
            raise RepresentationLadderError(
                "unsupported formal trained-representation receipt"
            )
        if self.representation_id not in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            raise RepresentationLadderError(
                "formal trained receipt applies only to R5/R5L"
            )
        for name in (
            "representation_coordinate_digest",
            "checkpoint_artifact_digest",
            "checkpoint_manifest_digest",
            "training_request_digest",
            "representation_execution_plan_digest",
            "formal_source_fit_batch_digest",
            "formal_trainer_contract_digest",
            "formal_fit_job_digest",
            "formal_source_fit_schedule_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected = sha256_json(self._payload_without_digest())
        if self.receipt_digest is None:
            object.__setattr__(self, "receipt_digest", expected)
        elif _digest(self.receipt_digest, "receipt_digest") != expected:
            raise RepresentationLadderError(
                "formal trained-representation receipt digest mismatch"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_id": self.representation_id,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "checkpoint_artifact_digest": self.checkpoint_artifact_digest,
            "checkpoint_manifest_digest": self.checkpoint_manifest_digest,
            "training_request_digest": self.training_request_digest,
            "representation_execution_plan_digest": (
                self.representation_execution_plan_digest
            ),
            "formal_source_fit_batch_digest": self.formal_source_fit_batch_digest,
            "formal_trainer_contract_digest": self.formal_trainer_contract_digest,
            "formal_fit_job_digest": self.formal_fit_job_digest,
            "formal_source_fit_schedule_digest": (
                self.formal_source_fit_schedule_digest
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    def validate_manifest(self, manifest: RepresentationManifest) -> None:
        if not isinstance(manifest, RepresentationManifest):
            raise RepresentationLadderError(
                "formal trained receipt requires RepresentationManifest"
            )
        if (
            manifest.representation_id != self.representation_id
            or manifest.coordinate_digest != self.representation_coordinate_digest
            or manifest.checkpoint_digest != self.checkpoint_artifact_digest
            or manifest.protocol_digest != self.training_request_digest
        ):
            raise RepresentationLadderError(
                "representation manifest differs from formal checkpoint receipt"
            )


@dataclass(frozen=True)
class RepresentationOutput:
    values: np.ndarray
    input_batch_digest: str
    coordinate_digest: str
    output_digest: str | None = None
    schema: str = REPRESENTATION_OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTATION_OUTPUT_SCHEMA:
            raise RepresentationLadderError("unsupported representation output schema")
        object.__setattr__(self, "values", _finite_matrix(self.values, "output values"))
        for name in ("input_batch_digest", "coordinate_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected = sha256_json(
            {
                "schema": self.schema,
                "input_batch_digest": self.input_batch_digest,
                "coordinate_digest": self.coordinate_digest,
                "arrays_digest": sha256_ndarrays({"values": self.values}),
            }
        )
        if self.output_digest is None:
            object.__setattr__(self, "output_digest", expected)
        elif _digest(self.output_digest, "output_digest") != expected:
            raise RepresentationLadderError("output_digest does not match values")


@dataclass(frozen=True)
class FittedRepresentation:
    """Runtime adapter whose callable is excluded from persisted identity."""

    manifest: RepresentationManifest
    _transform: Callable[[np.ndarray], np.ndarray] = field(repr=False, compare=False)
    checkpoint_bytes: bytes | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RepresentationManifest):
            raise RepresentationLadderError("manifest must be typed")
        if not callable(self._transform):
            raise RepresentationLadderError("transform must be callable")
        if self.checkpoint_bytes is not None:
            if not isinstance(self.checkpoint_bytes, bytes) or not self.checkpoint_bytes:
                raise RepresentationLadderError(
                    "checkpoint_bytes must be non-empty immutable bytes"
                )
            if self.manifest.checkpoint_digest != sha256_bytes(self.checkpoint_bytes):
                raise RepresentationLadderError(
                    "checkpoint_bytes do not match the representation manifest"
                )

    def transform(self, batch: RepresentationBatch) -> RepresentationOutput:
        if not isinstance(batch, RepresentationBatch):
            raise RepresentationLadderError("transform batch must be typed")
        if batch.input_dim != self.manifest.input_dim:
            raise RepresentationLadderError(
                "query input dimension differs from fitted representation"
            )
        values = _finite_matrix(self._transform(batch.values), "transformed values")
        if values.shape != (batch.values.shape[0], self.manifest.output_dim):
            raise RepresentationLadderError("transform returned an incompatible shape")
        return RepresentationOutput(
            values=values,
            input_batch_digest=batch.batch_digest,
            coordinate_digest=str(self.manifest.coordinate_digest),
        )


def _source(source: RepresentationBatch) -> RepresentationBatch:
    if not isinstance(source, RepresentationBatch) or source.role != "SOURCE_FIT":
        raise RepresentationLadderError("fit accepts only role=SOURCE_FIT")
    return source


def _implementation(representation_id: str, version: str) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v03-representation-implementation.v0",
            "representation_id": representation_id,
            "version": version,
            "numeric_backend": "numpy-float64",
        }
    )


def _checkpoint(representation_id: str, params_digest: str) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v03-representation-checkpoint.v0",
            "representation_id": representation_id,
            "params_digest": params_digest,
        }
    )


def _manifest(
    *,
    representation_id: str,
    source: RepresentationBatch,
    output_dim: int,
    protocol: Mapping[str, Any],
    params_digest: str,
    checkpoint_digest: str | None,
    implementation_digest: str,
    seed: int | None,
    source_fit_digest: str | None = None,
) -> RepresentationManifest:
    return RepresentationManifest(
        representation_id=representation_id,
        architecture=ARCHITECTURES[representation_id],
        input_dim=source.input_dim,
        output_dim=output_dim,
        protocol_digest=sha256_json(dict(protocol)),
        params_digest=params_digest,
        source_fit_digest=source.batch_digest if source_fit_digest is None else source_fit_digest,
        implementation_digest=implementation_digest,
        checkpoint_digest=checkpoint_digest,
        seed=seed,
    )


def fit_r0_identity(source: RepresentationBatch) -> FittedRepresentation:
    source = _source(source)
    params = sha256_json(
        {"schema": "policy-learnware.v03-r0-identity-params.v0", "parameters": []}
    )
    manifest = _manifest(
        representation_id=R0_PADDED_RAW,
        source=source,
        output_dim=source.input_dim,
        protocol={
            "schema": "policy-learnware.v03-r0-identity-protocol.v0",
            "operation": "identity",
        },
        params_digest=params,
        checkpoint_digest=None,
        implementation_digest=_implementation(R0_PADDED_RAW, "identity-copy/v0"),
        seed=None,
    )
    return FittedRepresentation(manifest, lambda values: np.array(values, copy=True))


def fit_r1_random_linear(
    source: RepresentationBatch, *, output_dim: int, seed: int
) -> FittedRepresentation:
    source = _source(source)
    output_dim = _positive_int(output_dim, "output_dim")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(source.input_dim)
    matrix = rng.normal(0.0, scale, size=(source.input_dim, output_dim)).astype(
        np.float64
    )
    params = sha256_ndarrays({"matrix": matrix})
    manifest = _manifest(
        representation_id=R1_FIXED_RANDOM_LINEAR,
        source=source,
        output_dim=output_dim,
        protocol={
            "schema": "policy-learnware.v03-r1-random-linear-protocol.v0",
            "distribution": "normal(0,1/sqrt(input_dim))",
            "bias": False,
            "nonlinearity": None,
            "input_dim": source.input_dim,
            "output_dim": output_dim,
            "seed": seed,
        },
        params_digest=params,
        checkpoint_digest=_checkpoint(R1_FIXED_RANDOM_LINEAR, params),
        implementation_digest=_implementation(
            R1_FIXED_RANDOM_LINEAR, "numpy-pcg64-matmul/v0"
        ),
        seed=seed,
    )
    matrix.setflags(write=False)
    return FittedRepresentation(manifest, lambda values: values @ matrix)


def fit_r2_pca_whitening(
    source: RepresentationBatch,
    *,
    output_dim: int,
    whiten: bool = True,
    epsilon: float = 1.0e-12,
) -> FittedRepresentation:
    source = _source(source)
    output_dim = _positive_int(output_dim, "output_dim")
    maximum = min(source.values.shape)
    if output_dim > maximum:
        raise RepresentationLadderError(
            "PCA output_dim cannot exceed min(source rows, input_dim)"
        )
    if isinstance(whiten, (bool, np.bool_)) is False:
        raise RepresentationLadderError("whiten must be boolean")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise RepresentationLadderError("epsilon must be finite and positive")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise RepresentationLadderError("epsilon must be finite and positive")

    mean = np.mean(source.values, axis=0)
    centered = source.values - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.array(vt[:output_dim], dtype=np.float64, copy=True)
    # Resolve the otherwise arbitrary sign of each singular vector using the
    # largest-magnitude loading and a positive-tie convention.
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0.0:
            components[index] *= -1.0
    if whiten:
        denominator = singular[:output_dim]
        scale = np.where(
            denominator > epsilon,
            math.sqrt(max(1, source.values.shape[0] - 1)) / denominator,
            0.0,
        )
    else:
        scale = np.ones(output_dim, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    params = sha256_ndarrays(
        {"mean": mean, "components": components, "scale": scale}
    )
    manifest = _manifest(
        representation_id=R2_SOURCE_PCA_WHITEN,
        source=source,
        output_dim=output_dim,
        protocol={
            "schema": "policy-learnware.v03-r2-pca-protocol.v0",
            "solver": "numpy.linalg.svd(full_matrices=false)",
            "sign_convention": "largest-absolute-loading-positive",
            "whiten": bool(whiten),
            "epsilon": epsilon,
            "output_dim": output_dim,
        },
        params_digest=params,
        checkpoint_digest=_checkpoint(R2_SOURCE_PCA_WHITEN, params),
        implementation_digest=_implementation(
            R2_SOURCE_PCA_WHITEN, "source-only-svd-fixed-sign/v0"
        ),
        seed=None,
    )
    for array in (mean, components, scale):
        array.setflags(write=False)
    return FittedRepresentation(
        manifest,
        lambda values: ((values - mean) @ components.T) * scale,
    )


def _random_dense(
    rng: np.random.Generator, input_dim: int, output_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    scale = math.sqrt(2.0 / input_dim)
    weight = rng.normal(0.0, scale, size=(input_dim, output_dim)).astype(np.float64)
    bias = np.zeros(output_dim, dtype=np.float64)
    return weight, bias


def fit_r3_matched_random_mlp(
    source: RepresentationBatch,
    *,
    output_dim: int = 32,
    hidden_dims: tuple[int, int] = (256, 256),
    seed: int,
) -> FittedRepresentation:
    source = _source(source)
    output_dim = _positive_int(output_dim, "output_dim")
    if (
        not isinstance(hidden_dims, tuple)
        or len(hidden_dims) != 2
        or any(_positive_int(item, "hidden_dims[]") <= 0 for item in hidden_dims)
    ):
        raise RepresentationLadderError("R3 requires exactly two positive hidden widths")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    w1, b1 = _random_dense(rng, source.input_dim, hidden_dims[0])
    w2, b2 = _random_dense(rng, hidden_dims[0], hidden_dims[1])
    w3, b3 = _random_dense(rng, hidden_dims[1], output_dim)
    arrays = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}
    params = sha256_ndarrays(arrays)
    manifest = _manifest(
        representation_id=R3_MATCHED_RANDOM_MLP,
        source=source,
        output_dim=output_dim,
        protocol={
            "schema": "policy-learnware.v03-r3-random-mlp-protocol.v0",
            "hidden_dims": list(hidden_dims),
            "activation": "relu",
            "l2_normalize_output": True,
            "initializer": "he-normal-zero-bias",
            "seed": seed,
        },
        params_digest=params,
        checkpoint_digest=_checkpoint(R3_MATCHED_RANDOM_MLP, params),
        implementation_digest=_implementation(
            R3_MATCHED_RANDOM_MLP, "two-hidden-relu-l2/v0"
        ),
        seed=seed,
    )
    for array in arrays.values():
        array.setflags(write=False)

    def transform(values: np.ndarray) -> np.ndarray:
        hidden1 = np.maximum(values @ w1 + b1, 0.0)
        hidden2 = np.maximum(hidden1 @ w2 + b2, 0.0)
        return _l2_normalize(hidden2 @ w3 + b3)

    return FittedRepresentation(manifest, transform)


def bind_historical_random_tanh(
    source: RepresentationBatch,
    *,
    spec: HistoricalRandomTanhSpec,
) -> FittedRepresentation:
    """Adapt the one canonical historical control to the representation API.

    Parameter generation, protocol identity and checkpoint identity are owned
    by :class:`HistoricalRandomTanhSpec`; this function must not create a second
    numerical or digest recipe for the same 14th control.
    """

    source = _source(source)
    if not isinstance(spec, HistoricalRandomTanhSpec):
        raise RepresentationLadderError(
            "historical control must use HistoricalRandomTanhSpec"
        )
    if spec.input_dim != source.input_dim:
        raise RepresentationLadderError("historical spec input dimension differs")
    implementation_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-representation-implementation.v0",
            "representation_id": R_HIST_RANDOM_TANH,
            "version": "canonical-historical-random-view-adapter/v0",
            "numeric_backend": "numpy-float32",
        }
    )
    manifest = RepresentationManifest(
        representation_id=R_HIST_RANDOM_TANH,
        architecture=ARCHITECTURES[R_HIST_RANDOM_TANH],
        input_dim=source.input_dim,
        output_dim=spec.output_dim,
        protocol_digest=str(spec.representation_protocol_digest),
        params_digest=spec.parameter_digest,
        source_fit_digest=source.batch_digest,
        implementation_digest=implementation_digest,
        checkpoint_digest=str(spec.checkpoint_digest),
        seed=spec.seed,
    )

    def transform(values: np.ndarray) -> np.ndarray:
        packed = np.asarray(values, dtype=np.float32)
        return np.asarray(
            np.tanh(packed @ spec.matrix + spec.bias), dtype=np.float32
        )

    return FittedRepresentation(manifest, transform)


@dataclass(frozen=True)
class TrainingRequest:
    representation_id: str
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    activation: str | None
    l2_normalize_output: bool
    objective_digest: str
    seed: int

    def __post_init__(self) -> None:
        if self.representation_id not in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            raise RepresentationLadderError("unsupported trainable representation")
        object.__setattr__(self, "input_dim", _positive_int(self.input_dim, "input_dim"))
        object.__setattr__(
            self, "output_dim", _positive_int(self.output_dim, "output_dim")
        )
        hidden = tuple(_positive_int(item, "hidden_dims[]") for item in self.hidden_dims)
        if self.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
            if len(hidden) != 2 or self.activation != "relu":
                raise RepresentationLadderError("R5 requires a two-hidden-layer ReLU MLP")
        elif hidden or self.activation is not None:
            raise RepresentationLadderError("R5L must be a single linear projection")
        if self.l2_normalize_output is not True:
            raise RepresentationLadderError("R5/R5L outputs must be L2-normalized")
        object.__setattr__(self, "hidden_dims", hidden)
        object.__setattr__(
            self, "objective_digest", _digest(self.objective_digest, "objective_digest")
        )
        object.__setattr__(self, "seed", _seed(self.seed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_REQUEST_SCHEMA,
            "representation_id": self.representation_id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "l2_normalize_output": self.l2_normalize_output,
            "objective_digest": self.objective_digest,
            "seed": self.seed,
        }

    @property
    def request_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRequest":
        fields = {
            "schema",
            "representation_id",
            "input_dim",
            "output_dim",
            "hidden_dims",
            "activation",
            "l2_normalize_output",
            "objective_digest",
            "seed",
        }
        _strict(value, fields, "training request")
        if value["schema"] != TRAINING_REQUEST_SCHEMA:
            raise RepresentationLadderError("unsupported training request schema")
        return cls(
            representation_id=value["representation_id"],
            input_dim=value["input_dim"],
            output_dim=value["output_dim"],
            hidden_dims=tuple(value["hidden_dims"]),
            activation=value["activation"],
            l2_normalize_output=value["l2_normalize_output"],
            objective_digest=value["objective_digest"],
            seed=value["seed"],
        )


@dataclass(frozen=True)
class TrainedCallableArtifact:
    """Output of an injected CPU/GPU trainer, with no authority semantics."""

    checkpoint_bytes: bytes
    parameter_digest: str
    trainer_implementation_digest: str
    transform: Callable[[np.ndarray], np.ndarray] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_bytes, bytes) or not self.checkpoint_bytes:
            raise RepresentationLadderError("trainer checkpoint_bytes cannot be empty")
        object.__setattr__(
            self, "parameter_digest", _digest(self.parameter_digest, "parameter_digest")
        )
        object.__setattr__(
            self,
            "trainer_implementation_digest",
            _digest(self.trainer_implementation_digest, "trainer_implementation_digest"),
        )
        if not callable(self.transform):
            raise RepresentationLadderError("trainer transform must be callable")


class RepresentationTrainer(Protocol):
    def __call__(
        self,
        source_values: np.ndarray,
        labels: np.ndarray,
        request: TrainingRequest,
    ) -> TrainedCallableArtifact: ...


@runtime_checkable
class RepresentationRestorer(Protocol):
    def restore(
        self,
        checkpoint_bytes: bytes,
        request: TrainingRequest,
    ) -> TrainedCallableArtifact: ...


def _categorical_labels(value: Any, rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (rows,) or raw.dtype.kind not in "iuUS":
        raise RepresentationLadderError(
            "source categorical labels must be a one-dimensional integer/string array"
        )
    if raw.dtype.kind in "US" and any(not str(item) for item in raw.tolist()):
        raise RepresentationLadderError("source categorical labels cannot be empty")
    result = np.ascontiguousarray(raw).copy()
    result.setflags(write=False)
    return result


def _fit_injected(
    source: RepresentationBatch,
    labels: np.ndarray,
    trainer: RepresentationTrainer,
    request: TrainingRequest,
) -> FittedRepresentation:
    source = _source(source)
    labels = _categorical_labels(labels, source.values.shape[0])
    if not callable(trainer):
        raise RepresentationLadderError("trainer must be callable")
    artifact = trainer(source.values, labels, request)
    if not isinstance(artifact, TrainedCallableArtifact):
        raise RepresentationLadderError("trainer returned an invalid artifact")
    # A source-only smoke validates shape, finiteness, and deterministic frozen
    # inference without exposing any query batch to the trainer.
    first = _finite_matrix(artifact.transform(source.values), "trainer source output")
    second = _finite_matrix(artifact.transform(source.values), "trainer source output")
    expected_shape = (source.values.shape[0], request.output_dim)
    if first.shape != expected_shape or not np.array_equal(first, second):
        raise RepresentationLadderError(
            "trainer must return a deterministic transform with the requested shape"
        )
    label_digest = sha256_ndarrays({"labels": labels})
    source_fit_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-supervised-source-fit.v0",
            "source_batch_digest": source.batch_digest,
            "label_digest": label_digest,
            "training_request_digest": request.request_digest,
        }
    )
    params = sha256_json(
        {
            "schema": "policy-learnware.v03-trained-parameter-binding.v0",
            "trainer_parameter_digest": artifact.parameter_digest,
            "training_request_digest": request.request_digest,
        }
    )
    checkpoint_digest = sha256_bytes(artifact.checkpoint_bytes)
    manifest = _manifest(
        representation_id=request.representation_id,
        source=source,
        output_dim=request.output_dim,
        protocol=request.to_dict(),
        params_digest=params,
        checkpoint_digest=checkpoint_digest,
        implementation_digest=artifact.trainer_implementation_digest,
        seed=request.seed,
        source_fit_digest=source_fit_digest,
    )
    return FittedRepresentation(
        manifest, artifact.transform, checkpoint_bytes=artifact.checkpoint_bytes
    )


def restore_trained_representation(
    *,
    manifest: RepresentationManifest,
    checkpoint_bytes: bytes,
    request: TrainingRequest,
    restorer: RepresentationRestorer,
    verification_source: RepresentationBatch,
    labels: np.ndarray,
) -> FittedRepresentation:
    """Restore one R5/R5L transform from byte-verified source-only state."""

    source = _source(verification_source)
    if not isinstance(manifest, RepresentationManifest):
        raise RepresentationLadderError("restore requires RepresentationManifest")
    if not isinstance(request, TrainingRequest):
        raise RepresentationLadderError("restore requires TrainingRequest")
    if manifest.representation_id != request.representation_id:
        raise RepresentationLadderError("restore request representation differs")
    if manifest.protocol_digest != request.request_digest:
        raise RepresentationLadderError("restore request protocol differs from manifest")
    if manifest.input_dim != source.input_dim or manifest.input_dim != request.input_dim:
        raise RepresentationLadderError("restore source input dimension differs")
    if manifest.output_dim != request.output_dim:
        raise RepresentationLadderError("restore output dimension differs")
    if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
        raise RepresentationLadderError("restore checkpoint_bytes must be non-empty")
    if sha256_bytes(checkpoint_bytes) != manifest.checkpoint_digest:
        raise RepresentationLadderError("restore checkpoint bytes differ from manifest")
    if not isinstance(restorer, RepresentationRestorer):
        raise RepresentationLadderError("restorer does not implement restore contract")
    categorical = _categorical_labels(labels, source.values.shape[0])
    label_digest = sha256_ndarrays({"labels": categorical})
    expected_source_fit = sha256_json(
        {
            "schema": "policy-learnware.v03-supervised-source-fit.v0",
            "source_batch_digest": source.batch_digest,
            "label_digest": label_digest,
            "training_request_digest": request.request_digest,
        }
    )
    if manifest.source_fit_digest != expected_source_fit:
        raise RepresentationLadderError(
            "restore verification source/labels differ from fitted manifest"
        )
    artifact = restorer.restore(checkpoint_bytes, request)
    if not isinstance(artifact, TrainedCallableArtifact):
        raise RepresentationLadderError("restorer returned an invalid artifact")
    if artifact.checkpoint_bytes != checkpoint_bytes:
        raise RepresentationLadderError("restorer changed checkpoint bytes")
    expected_params = sha256_json(
        {
            "schema": "policy-learnware.v03-trained-parameter-binding.v0",
            "trainer_parameter_digest": artifact.parameter_digest,
            "training_request_digest": request.request_digest,
        }
    )
    if expected_params != manifest.params_digest:
        raise RepresentationLadderError("restored parameters differ from manifest")
    if artifact.trainer_implementation_digest != manifest.implementation_digest:
        raise RepresentationLadderError("restored implementation differs from manifest")
    first = _finite_matrix(artifact.transform(source.values), "restored source output")
    second = _finite_matrix(artifact.transform(source.values), "restored source output")
    if first.shape != (source.values.shape[0], request.output_dim) or not np.array_equal(
        first, second
    ):
        raise RepresentationLadderError(
            "restored transform must be deterministic and match output_dim"
        )
    return FittedRepresentation(
        manifest, artifact.transform, checkpoint_bytes=checkpoint_bytes
    )


def fit_r5_corro_style(
    source: RepresentationBatch,
    *,
    labels: np.ndarray,
    trainer: RepresentationTrainer,
    objective_digest: str,
    seed: int,
    output_dim: int = 32,
    hidden_dims: tuple[int, int] = (256, 256),
) -> FittedRepresentation:
    source = _source(source)
    request = TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=source.input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        l2_normalize_output=True,
        objective_digest=objective_digest,
        seed=seed,
    )
    return _fit_injected(source, labels, trainer, request)


def fit_r5l_supervised_linear(
    source: RepresentationBatch,
    *,
    labels: np.ndarray,
    trainer: RepresentationTrainer,
    objective_digest: str,
    seed: int,
    output_dim: int = 32,
) -> FittedRepresentation:
    source = _source(source)
    request = TrainingRequest(
        representation_id=R5L_SUPERVISED_LINEAR,
        input_dim=source.input_dim,
        output_dim=output_dim,
        hidden_dims=(),
        activation=None,
        l2_normalize_output=True,
        objective_digest=objective_digest,
        seed=seed,
    )
    return _fit_injected(source, labels, trainer, request)


def bind_r4_frozen_callable(
    source: RepresentationBatch,
    *,
    output_dim: int,
    checkpoint_digest: str,
    normalizer_digest: str,
    implementation_digest: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> FittedRepresentation:
    """Bind an archived CORRO-style callable without retraining it."""

    source = _source(source)
    output_dim = _positive_int(output_dim, "output_dim")
    checkpoint_digest = _digest(checkpoint_digest, "checkpoint_digest")
    normalizer_digest = _digest(normalizer_digest, "normalizer_digest")
    implementation_digest = _digest(implementation_digest, "implementation_digest")
    if not callable(transform):
        raise RepresentationLadderError("frozen transform must be callable")
    first = _finite_matrix(transform(source.values), "frozen source output")
    second = _finite_matrix(transform(source.values), "frozen source output")
    if first.shape != (source.values.shape[0], output_dim) or not np.array_equal(
        first, second
    ):
        raise RepresentationLadderError(
            "frozen transform must be deterministic and match output_dim"
        )
    params = sha256_json(
        {
            "schema": "policy-learnware.v03-r4-frozen-params.v0",
            "normalizer_digest": normalizer_digest,
        }
    )
    manifest = _manifest(
        representation_id=R4_ARCHIVED_FROZEN_CORRO,
        source=source,
        output_dim=output_dim,
        protocol={
            "schema": "policy-learnware.v03-r4-frozen-protocol.v0",
            "architecture": ARCHITECTURES[R4_ARCHIVED_FROZEN_CORRO],
            "normalizer_digest": normalizer_digest,
            "checkpoint_digest": checkpoint_digest,
        },
        params_digest=params,
        checkpoint_digest=checkpoint_digest,
        implementation_digest=implementation_digest,
        seed=None,
    )
    return FittedRepresentation(manifest, transform)


__all__ = [
    "ARCHITECTURES",
    "FittedRepresentation",
    "FORMAL_TRAINED_REPRESENTATION_RECEIPT_SCHEMA",
    "FormalTrainedRepresentationReceipt",
    "R0_PADDED_RAW",
    "R1_FIXED_RANDOM_LINEAR",
    "R2_SOURCE_PCA_WHITEN",
    "R3_MATCHED_RANDOM_MLP",
    "R4_ARCHIVED_FROZEN_CORRO",
    "R5L_SUPERVISED_LINEAR",
    "R5_VIEW_SPECIFIC_CORRO_REFIT",
    "REPRESENTATION_IDS",
    "R_HIST_RANDOM_TANH",
    "RepresentationBatch",
    "RepresentationLadderError",
    "RepresentationManifest",
    "RepresentationOutput",
    "RepresentationTrainer",
    "RepresentationRestorer",
    "TrainedCallableArtifact",
    "TrainingRequest",
    "bind_historical_random_tanh",
    "bind_r4_frozen_callable",
    "fit_r0_identity",
    "fit_r1_random_linear",
    "fit_r2_pca_whitening",
    "fit_r3_matched_random_mlp",
    "fit_r5_corro_style",
    "fit_r5l_supervised_linear",
    "restore_trained_representation",
]
