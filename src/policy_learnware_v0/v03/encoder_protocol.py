"""Encoder-neutral fit/freeze/encode contracts for the v0.3 bake-off."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from .access import (
    PROBE_STYLE_LABELS,
    SOURCE_ANCHOR_CATEGORICAL_LABELS,
    SOURCE_AXIS_CATEGORICAL_LABELS,
    SOURCE_NUMERIC_FACTOR_PARAMETERS,
    SOURCE_TASK_CATEGORICAL_LABELS,
    EncoderAccessCard,
    REWARD_CHANNEL,
    SOURCE_EPISODE_PROBE_BOUNDARIES,
    SOURCE_RAW_TRANSITIONS,
)
from .data_roles import (
    DataRoleManifest,
    DataRoleRecord,
    validate_loto_isolation,
)
from .schemas import (
    EncoderProtocolRecord,
    LOTOFoldRecord,
    checked_digest,
    checked_safe_id,
)
from .windowing import TransitionWindowBatch


class EncoderProtocolError(ValueError):
    """An encoder call violates shape, digest, access, or lifecycle constraints."""


_SANITIZED_WINDOW_ID = re.compile(r"^v03w-[0-9a-f]{32}$")
_INPUT_CHANNEL_CAPABILITY = {
    "observation": SOURCE_RAW_TRANSITIONS,
    "action": SOURCE_RAW_TRANSITIONS,
    "reward": REWARD_CHANNEL,
    "next_observation": SOURCE_RAW_TRANSITIONS,
    "terminated": SOURCE_RAW_TRANSITIONS,
    "truncated": SOURCE_RAW_TRANSITIONS,
    "observation_mask": SOURCE_RAW_TRANSITIONS,
    "action_mask": SOURCE_RAW_TRANSITIONS,
}
ENCODER_INPUT_CHANNELS = frozenset(_INPUT_CHANNEL_CAPABILITY)
TargetQueryRole = Literal["development_query", "confirmatory_query"]


def _derive_sanitized_window_ids(
    channels: Mapping[str, np.ndarray],
    window_mask: np.ndarray,
    *,
    input_view_digest: str,
    window_protocol_digest: str,
) -> tuple[str, ...]:
    identifiers: list[str] = []
    for row_index in range(window_mask.shape[0]):
        visible_row = {
            name: array[row_index] for name, array in channels.items()
        }
        visible_row["window_mask"] = window_mask[row_index]
        material = {
            "schema": "policy-learnware.v03-sanitized-window-id.v0",
            "row_index": row_index,
            "visible_arrays_digest": sha256_ndarrays(visible_row),
            "input_view_digest": input_view_digest,
            "window_protocol_digest": window_protocol_digest,
        }
        identifiers.append(f"v03w-{sha256_json(material)[:32]}")
    return tuple(identifiers)


@dataclass(frozen=True)
class CostRecord:
    wall_seconds: float
    peak_memory_bytes: int
    device: str
    trial_count: int

    def __post_init__(self) -> None:
        if isinstance(self.wall_seconds, bool) or not isinstance(
            self.wall_seconds, (int, float)
        ):
            raise EncoderProtocolError("wall_seconds must be finite and non-negative")
        seconds = float(self.wall_seconds)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise EncoderProtocolError("wall_seconds must be finite and non-negative")
        object.__setattr__(self, "wall_seconds", seconds)
        for name in ("peak_memory_bytes", "trial_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EncoderProtocolError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "device", checked_safe_id(self.device, "device"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.wall_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "device": self.device,
            "trial_count": self.trial_count,
        }


@dataclass(frozen=True)
class SanitizedEncoderInputBatch:
    """The complete and only transition surface visible to an encoder.

    It deliberately has no pointer back to ``CanonicalTransitionBatch`` or
    ``TransitionWindowBatch``.  Channel arrays, the temporal validity mask and
    opaque IDs are copied into a closed, immutable value object.  Consequently
    an adapter cannot discover an omitted reward channel or private episode/task
    metadata through object traversal.
    """

    channels: Mapping[str, np.ndarray]
    window_mask: np.ndarray
    window_ids: tuple[str, ...]
    input_view_digest: str
    window_protocol_digest: str
    access_card_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.channels, Mapping) or not self.channels:
            raise EncoderProtocolError("sanitized encoder channels must be non-empty")
        unknown = set(self.channels) - set(ENCODER_INPUT_CHANNELS)
        if unknown or not all(isinstance(name, str) for name in self.channels):
            raise EncoderProtocolError(
                f"unknown sanitized encoder channels: {sorted(str(x) for x in unknown)}"
            )
        raw_mask = np.asarray(self.window_mask)
        if (
            raw_mask.dtype.kind != "b"
            or raw_mask.ndim != 2
            or any(size <= 0 for size in raw_mask.shape)
        ):
            raise EncoderProtocolError(
                "sanitized window_mask must be a non-empty boolean matrix"
            )
        mask = np.array(raw_mask, dtype=np.bool_, copy=True)
        for row in mask:
            valid_count = int(np.sum(row))
            if valid_count <= 0 or not np.all(row[:valid_count]) or np.any(
                row[valid_count:]
            ):
                raise EncoderProtocolError(
                    "sanitized window_mask rows must be non-empty true prefixes"
                )

        frozen: dict[str, np.ndarray] = {}
        for name, value in sorted(self.channels.items()):
            array = np.asarray(value)
            if array.dtype.hasobject or array.ndim < 2 or array.shape[:2] != mask.shape:
                raise EncoderProtocolError(
                    f"sanitized channel {name!r} must align with [window, time]"
                )
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise EncoderProtocolError(
                    f"sanitized channel {name!r} must be finite"
                )
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            frozen[name] = copied

        input_view_digest = checked_digest(
            self.input_view_digest, "input_view_digest"
        )
        window_protocol_digest = checked_digest(
            self.window_protocol_digest, "window_protocol_digest"
        )
        access_card_digest = checked_digest(
            self.access_card_digest, "access_card_digest"
        )
        identifiers = tuple(self.window_ids)
        if (
            len(identifiers) != mask.shape[0]
            or len(set(identifiers)) != len(identifiers)
            or any(
                not isinstance(item, str) or not _SANITIZED_WINDOW_ID.fullmatch(item)
                for item in identifiers
            )
        ):
            raise EncoderProtocolError(
                "sanitized window_ids must be unique opaque IDs aligned to rows"
            )
        expected_ids = _derive_sanitized_window_ids(
            frozen,
            mask,
            input_view_digest=input_view_digest,
            window_protocol_digest=window_protocol_digest,
        )
        if identifiers != expected_ids:
            raise EncoderProtocolError(
                "sanitized window_ids differ from visible-channel derivation"
            )
        mask.setflags(write=False)
        object.__setattr__(self, "channels", MappingProxyType(frozen))
        object.__setattr__(self, "window_mask", mask)
        object.__setattr__(self, "window_ids", identifiers)
        object.__setattr__(self, "input_view_digest", input_view_digest)
        object.__setattr__(self, "window_protocol_digest", window_protocol_digest)
        object.__setattr__(self, "access_card_digest", access_card_digest)

    @property
    def channel_allowlist(self) -> tuple[str, ...]:
        return tuple(self.channels)

    @property
    def window_length(self) -> int:
        return int(self.window_mask.shape[1])

    @property
    def sanitized_input_digest(self) -> str:
        arrays = {**self.channels, "window_mask": self.window_mask}
        return sha256_json(
            {
                "schema": "policy-learnware.v03-sanitized-encoder-input.v0",
                "channel_allowlist": list(self.channel_allowlist),
                "arrays_digest": sha256_ndarrays(arrays),
                "window_ids": list(self.window_ids),
                "input_view_digest": self.input_view_digest,
                "window_protocol_digest": self.window_protocol_digest,
                "access_card_digest": self.access_card_digest,
            }
        )

    def channel(self, name: str) -> np.ndarray:
        try:
            return self.channels[name]
        except KeyError as exc:
            raise EncoderProtocolError(
                f"encoder input channel {name!r} is not visible"
            ) from exc


def sanitize_encoder_inputs(
    windows: TransitionWindowBatch,
    *,
    access_card: EncoderAccessCard,
    channel_allowlist: Sequence[str],
    input_view_digest: str,
) -> SanitizedEncoderInputBatch:
    """Materialize an access-checked, identity-sanitized adapter input batch."""

    if not isinstance(windows, TransitionWindowBatch):
        raise EncoderProtocolError("raw encoder input must be TransitionWindowBatch")
    if not isinstance(access_card, EncoderAccessCard):
        raise EncoderProtocolError("access_card must be EncoderAccessCard")
    if isinstance(channel_allowlist, (str, bytes)):
        raise EncoderProtocolError("channel_allowlist must be a sequence of channel names")
    names = tuple(channel_allowlist)
    if not names or len(set(names)) != len(names) or not all(
        isinstance(name, str) for name in names
    ):
        raise EncoderProtocolError(
            "channel_allowlist must contain unique channel names"
        )
    unknown = set(names) - set(ENCODER_INPUT_CHANNELS)
    if unknown:
        raise EncoderProtocolError(f"unknown encoder input channels: {sorted(unknown)}")

    # Episode boundaries are consumed by the frozen windowing step and are
    # therefore an explicit capability even though raw episode IDs never cross
    # the sanitized boundary.
    access_card.assert_can_read(
        SOURCE_RAW_TRANSITIONS, SOURCE_EPISODE_PROBE_BOUNDARIES
    )
    access_card.assert_can_read(*(_INPUT_CHANNEL_CAPABILITY[name] for name in names))
    view_digest = checked_digest(input_view_digest, "input_view_digest")
    materialized = {
        name: windows.materialize_channel(name) for name in sorted(names)
    }
    mask = np.array(windows.window_mask, copy=True)

    # Do not reuse raw window IDs: their source transition digest may contain an
    # omitted channel (notably reward).  These IDs are derived solely from what
    # the adapter can actually see plus stable row order.
    identifiers = _derive_sanitized_window_ids(
        materialized,
        mask,
        input_view_digest=view_digest,
        window_protocol_digest=windows.window_protocol_digest,
    )
    return SanitizedEncoderInputBatch(
        channels=materialized,
        window_mask=mask,
        window_ids=identifiers,
        input_view_digest=view_digest,
        window_protocol_digest=windows.window_protocol_digest,
        access_card_digest=access_card.access_card_digest,
    )


@dataclass(frozen=True)
class EncoderSupervisionBatch:
    labels: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.labels, Mapping):
            raise EncoderProtocolError("supervision labels must be a mapping")
        frozen: dict[str, np.ndarray] = {}
        expected_count: int | None = None
        for name, value in sorted(self.labels.items()):
            checked_safe_id(name, "supervision label name")
            array = np.asarray(value)
            if array.ndim < 1 or array.shape[0] <= 0 or array.dtype.hasobject:
                raise EncoderProtocolError(
                    f"supervision label {name!r} must be a non-empty non-object array"
                )
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise EncoderProtocolError(f"supervision label {name!r} must be finite")
            if expected_count is None:
                expected_count = int(array.shape[0])
            elif array.shape[0] != expected_count:
                raise EncoderProtocolError("supervision labels have inconsistent row counts")
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            frozen[name] = copied
        object.__setattr__(self, "labels", MappingProxyType(frozen))

    @property
    def row_count(self) -> int | None:
        if not self.labels:
            return None
        return int(next(iter(self.labels.values())).shape[0])

    @property
    def supervision_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-encoder-supervision.v0",
                "labels_digest": sha256_ndarrays(self.labels),
            }
        )


def encoder_dataset_digest(
    inputs: SanitizedEncoderInputBatch,
    supervision: EncoderSupervisionBatch | None,
    *,
    role: Literal["source_encoder_train", "source_encoder_validation"],
) -> str:
    if not isinstance(inputs, SanitizedEncoderInputBatch):
        raise EncoderProtocolError("encoder dataset requires sanitized inputs")
    if role not in {"source_encoder_train", "source_encoder_validation"}:
        raise EncoderProtocolError("encoder dataset has an invalid logical role")
    if supervision is not None and not isinstance(supervision, EncoderSupervisionBatch):
        raise EncoderProtocolError("supervision must be EncoderSupervisionBatch")
    return sha256_json(
        {
            "schema": "policy-learnware.v03-encoder-dataset.v0",
            "sanitized_input_digest": inputs.sanitized_input_digest,
            "supervision_digest": (
                supervision.supervision_digest if supervision is not None else None
            ),
        }
    )


def _validate_bound_encoder_data(
    *,
    inputs: SanitizedEncoderInputBatch,
    supervision: EncoderSupervisionBatch | None,
    role_record: DataRoleRecord,
    role_manifest: DataRoleManifest,
    loto_fold: LOTOFoldRecord,
    target_query_role: TargetQueryRole,
    required_role: Literal["source_encoder_train", "source_encoder_validation"],
) -> None:
    if not isinstance(inputs, SanitizedEncoderInputBatch):
        raise EncoderProtocolError("encoder data requires SanitizedEncoderInputBatch")
    if supervision is not None:
        if not isinstance(supervision, EncoderSupervisionBatch):
            raise EncoderProtocolError("supervision must be EncoderSupervisionBatch")
        if supervision.row_count != len(inputs.window_ids):
            raise EncoderProtocolError("supervision rows must align with encoder inputs")
    if not isinstance(role_record, DataRoleRecord) or role_record.role != required_role:
        raise EncoderProtocolError(
            f"encoder data requires a typed {required_role} role record"
        )
    if not isinstance(role_manifest, DataRoleManifest):
        raise EncoderProtocolError("encoder data requires typed DataRoleManifest")
    if not isinstance(loto_fold, LOTOFoldRecord):
        raise EncoderProtocolError("encoder data requires typed LOTOFoldRecord")
    if target_query_role not in {"development_query", "confirmatory_query"}:
        raise EncoderProtocolError("invalid LOTO target query role")
    registered = role_manifest.records_for(required_role)
    if len(registered) != 1 or registered[0].role_record_digest != role_record.role_record_digest:
        raise EncoderProtocolError(
            f"{required_role} must be the unique record in the bound role manifest"
        )
    expected_dataset = encoder_dataset_digest(
        inputs, supervision, role=required_role
    )
    if role_record.dataset_digest != expected_dataset:
        raise EncoderProtocolError(
            f"{required_role} dataset digest does not match sanitized data"
        )
    try:
        validate_loto_isolation(
            loto_fold, role_manifest, target_query_role=target_query_role
        )
    except ValueError as exc:
        raise EncoderProtocolError(f"LOTO data isolation failed: {exc}") from exc


@dataclass(frozen=True)
class EncoderTrainingData:
    inputs: SanitizedEncoderInputBatch
    supervision: EncoderSupervisionBatch | None
    role_record: DataRoleRecord
    role_manifest: DataRoleManifest
    loto_fold: LOTOFoldRecord
    target_query_role: TargetQueryRole

    def __post_init__(self) -> None:
        _validate_bound_encoder_data(
            inputs=self.inputs,
            supervision=self.supervision,
            role_record=self.role_record,
            role_manifest=self.role_manifest,
            loto_fold=self.loto_fold,
            target_query_role=self.target_query_role,
            required_role="source_encoder_train",
        )

    @property
    def dataset_digest(self) -> str:
        return encoder_dataset_digest(
            self.inputs, self.supervision, role="source_encoder_train"
        )


@dataclass(frozen=True)
class EncoderValidationData:
    inputs: SanitizedEncoderInputBatch
    supervision: EncoderSupervisionBatch | None
    role_record: DataRoleRecord
    role_manifest: DataRoleManifest
    loto_fold: LOTOFoldRecord
    target_query_role: TargetQueryRole

    def __post_init__(self) -> None:
        _validate_bound_encoder_data(
            inputs=self.inputs,
            supervision=self.supervision,
            role_record=self.role_record,
            role_manifest=self.role_manifest,
            loto_fold=self.loto_fold,
            target_query_role=self.target_query_role,
            required_role="source_encoder_validation",
        )

    @property
    def dataset_digest(self) -> str:
        return encoder_dataset_digest(
            self.inputs, self.supervision, role="source_encoder_validation"
        )


@dataclass(frozen=True)
class EncoderTrainingProtocolBinding:
    """Canonical declaration of the transition view used by ``fit``.

    ``input_view_digest`` alone is an opaque label and cannot prove that an
    adapter was trained on the channels declared by its protocol.  This record
    binds the view, windowing protocol, and exact channel surface into the
    training-protocol digest.  It contains no task, query, or data-split IDs.
    """

    input_view_digest: str
    window_protocol_digest: str
    channel_allowlist: tuple[str, ...]
    training_recipe_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_view_digest",
            checked_digest(self.input_view_digest, "input_view_digest"),
        )
        object.__setattr__(
            self,
            "window_protocol_digest",
            checked_digest(self.window_protocol_digest, "window_protocol_digest"),
        )
        object.__setattr__(
            self,
            "training_recipe_digest",
            checked_digest(self.training_recipe_digest, "training_recipe_digest"),
        )
        channels = tuple(self.channel_allowlist)
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(not isinstance(name, str) for name in channels)
        ):
            raise EncoderProtocolError(
                "training protocol channel_allowlist must contain unique channel names"
            )
        unknown = set(channels) - set(ENCODER_INPUT_CHANNELS)
        if unknown:
            raise EncoderProtocolError(
                f"training protocol contains unknown channels: {sorted(unknown)}"
            )
        object.__setattr__(self, "channel_allowlist", tuple(sorted(channels)))

    @property
    def training_protocol_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-encoder-training-protocol-binding.v0",
                "input_view_digest": self.input_view_digest,
                "window_protocol_digest": self.window_protocol_digest,
                "channel_allowlist": list(self.channel_allowlist),
                "training_recipe_digest": self.training_recipe_digest,
            }
        )


@dataclass(frozen=True)
class EncoderTrainingContract:
    protocol_record: EncoderProtocolRecord
    access_card: EncoderAccessCard
    training_protocol_binding: EncoderTrainingProtocolBinding
    role_manifest: DataRoleManifest
    loto_fold: LOTOFoldRecord
    target_query_role: TargetQueryRole
    semantic_output_protocol_digest: str
    runtime_digest: str
    execution_mode: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_record, EncoderProtocolRecord):
            raise EncoderProtocolError("protocol_record must be EncoderProtocolRecord")
        if not isinstance(self.access_card, EncoderAccessCard):
            raise EncoderProtocolError("access_card must be EncoderAccessCard")
        if self.protocol_record.encoder_id != self.access_card.encoder_id:
            raise EncoderProtocolError("protocol/access-card encoder IDs disagree")
        if self.protocol_record.access_card_digest != self.access_card.access_card_digest:
            raise EncoderProtocolError("protocol/access-card digest mismatch")
        if not isinstance(
            self.training_protocol_binding, EncoderTrainingProtocolBinding
        ):
            raise EncoderProtocolError(
                "training contract requires EncoderTrainingProtocolBinding"
            )
        binding = self.training_protocol_binding
        if (
            self.protocol_record.input_view_digest != binding.input_view_digest
            or self.protocol_record.window_protocol_digest
            != binding.window_protocol_digest
            or self.protocol_record.training_protocol_digest
            != binding.training_protocol_digest
        ):
            raise EncoderProtocolError(
                "encoder protocol record differs from training protocol binding"
            )
        self.access_card.assert_can_read(
            SOURCE_RAW_TRANSITIONS,
            SOURCE_EPISODE_PROBE_BOUNDARIES,
            *(
                _INPUT_CHANNEL_CAPABILITY[name]
                for name in binding.channel_allowlist
            ),
        )
        if not isinstance(self.role_manifest, DataRoleManifest):
            raise EncoderProtocolError("training contract requires typed DataRoleManifest")
        if not isinstance(self.loto_fold, LOTOFoldRecord):
            raise EncoderProtocolError("training contract requires typed LOTOFoldRecord")
        if self.target_query_role not in {"development_query", "confirmatory_query"}:
            raise EncoderProtocolError("invalid training-contract target query role")
        try:
            validate_loto_isolation(
                self.loto_fold,
                self.role_manifest,
                target_query_role=self.target_query_role,
            )
        except ValueError as exc:
            raise EncoderProtocolError(f"LOTO training contract failed: {exc}") from exc
        for name in ("semantic_output_protocol_digest", "runtime_digest"):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        object.__setattr__(
            self, "execution_mode", checked_safe_id(self.execution_mode, "execution_mode")
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EncoderProtocolError("seed must be a non-negative integer")

    @property
    def contract_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-encoder-training-contract.v0",
                "protocol_record_digest": self.protocol_record.protocol_record_digest,
                "access_card_digest": self.access_card.access_card_digest,
                "training_protocol_binding_digest": (
                    self.training_protocol_binding.training_protocol_digest
                ),
                "role_manifest_digest": self.role_manifest.manifest_digest,
                "loto_fold_record_digest": self.loto_fold.fold_record_digest,
                "target_query_role": self.target_query_role,
                "train_role_digest": self.role_manifest.role_digest(
                    "source_encoder_train"
                ),
                "validation_role_digest": self.role_manifest.role_digest(
                    "source_encoder_validation"
                ),
                "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
                "runtime_digest": self.runtime_digest,
                "execution_mode": self.execution_mode,
                "seed": self.seed,
            }
        )


@dataclass(frozen=True)
class EncoderInferenceContract:
    checkpoint_digest: str
    input_view_digest: str
    window_protocol_digest: str
    semantic_output_protocol_digest: str
    runtime_digest: str
    execution_mode: str
    mathematical_dtype: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_digest",
            "input_view_digest",
            "window_protocol_digest",
            "semantic_output_protocol_digest",
            "runtime_digest",
        ):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        object.__setattr__(
            self, "execution_mode", checked_safe_id(self.execution_mode, "execution_mode")
        )
        if self.mathematical_dtype != "float64":
            raise EncoderProtocolError("v0.3 semantic output mathematical dtype is float64")

    @property
    def contract_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-encoder-inference-contract.v0",
                "checkpoint_digest": self.checkpoint_digest,
                "input_view_digest": self.input_view_digest,
                "window_protocol_digest": self.window_protocol_digest,
                "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
                "runtime_digest": self.runtime_digest,
                "execution_mode": self.execution_mode,
                "mathematical_dtype": self.mathematical_dtype,
            }
        )


@dataclass(frozen=True)
class EncoderFitResult:
    encoder_id: str
    checkpoint_digest: str
    training_manifest_digest: str
    protocol_record_digest: str
    training_contract_digest: str
    access_card_digest: str
    input_view_digest: str
    semantic_output_protocol_digest: str
    runtime_digest: str
    fold_id: str
    seed: int
    latent_dim: int
    training_cost: CostRecord

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_id", checked_safe_id(self.encoder_id, "encoder_id"))
        for name in (
            "checkpoint_digest",
            "training_manifest_digest",
            "protocol_record_digest",
            "training_contract_digest",
            "access_card_digest",
            "input_view_digest",
            "semantic_output_protocol_digest",
            "runtime_digest",
        ):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        object.__setattr__(self, "fold_id", checked_safe_id(self.fold_id, "fold_id"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EncoderProtocolError("fit seed must be a non-negative integer")
        if isinstance(self.latent_dim, bool) or not isinstance(self.latent_dim, int) or self.latent_dim <= 0:
            raise EncoderProtocolError("latent_dim must be a positive integer")
        if not isinstance(self.training_cost, CostRecord):
            raise EncoderProtocolError("training_cost must be CostRecord")

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "checkpoint_digest": self.checkpoint_digest,
            "training_manifest_digest": self.training_manifest_digest,
            "protocol_record_digest": self.protocol_record_digest,
            "training_contract_digest": self.training_contract_digest,
            "access_card_digest": self.access_card_digest,
            "input_view_digest": self.input_view_digest,
            "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
            "runtime_digest": self.runtime_digest,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "latent_dim": self.latent_dim,
            "training_cost": self.training_cost.to_dict(),
        }


AdapterFitRole = Literal["source_encoder_train", "source_encoder_validation"]


@dataclass(frozen=True)
class AdapterFitData:
    """Task-identity-free data surface passed to an encoder adapter."""

    logical_role: AdapterFitRole
    inputs: SanitizedEncoderInputBatch
    supervision: EncoderSupervisionBatch | None
    dataset_digest: str

    def __post_init__(self) -> None:
        if self.logical_role not in {
            "source_encoder_train",
            "source_encoder_validation",
        }:
            raise EncoderProtocolError("adapter fit data has an invalid logical role")
        if not isinstance(self.inputs, SanitizedEncoderInputBatch):
            raise EncoderProtocolError("adapter fit data requires sanitized inputs")
        if self.supervision is not None:
            if not isinstance(self.supervision, EncoderSupervisionBatch):
                raise EncoderProtocolError(
                    "adapter fit supervision must be EncoderSupervisionBatch"
                )
            if self.supervision.row_count != len(self.inputs.window_ids):
                raise EncoderProtocolError(
                    "adapter fit supervision rows differ from inputs"
                )
        expected = encoder_dataset_digest(
            self.inputs, self.supervision, role=self.logical_role
        )
        observed = checked_digest(self.dataset_digest, "dataset_digest")
        if observed != expected:
            raise EncoderProtocolError(
                "adapter fit dataset digest differs from sanitized data"
            )
        object.__setattr__(self, "dataset_digest", observed)

    @property
    def adapter_data_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-adapter-fit-data.v0",
                "logical_role": self.logical_role,
                "dataset_digest": self.dataset_digest,
            }
        )


@dataclass(frozen=True)
class AdapterTrainingContract:
    """Sanitized fit contract with an opaque LOTO binding.

    The full role manifest and LOTO record stay on the framework side.  Only
    their commitment is exposed, so an adapter cannot inspect confirmatory
    task IDs, query roles, or query seed tokens.
    """

    protocol_record: EncoderProtocolRecord
    access_card: EncoderAccessCard
    training_protocol_binding: EncoderTrainingProtocolBinding
    training_contract_digest: str
    abstract_fold_id: str
    abstract_fold_digest: str
    training_dataset_digest: str
    validation_dataset_digest: str
    semantic_output_protocol_digest: str
    runtime_digest: str
    execution_mode: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_record, EncoderProtocolRecord):
            raise EncoderProtocolError(
                "adapter contract requires EncoderProtocolRecord"
            )
        if not isinstance(self.access_card, EncoderAccessCard):
            raise EncoderProtocolError("adapter contract requires EncoderAccessCard")
        if not isinstance(
            self.training_protocol_binding, EncoderTrainingProtocolBinding
        ):
            raise EncoderProtocolError(
                "adapter contract requires EncoderTrainingProtocolBinding"
            )
        if (
            self.protocol_record.encoder_id != self.access_card.encoder_id
            or self.protocol_record.access_card_digest
            != self.access_card.access_card_digest
            or self.protocol_record.input_view_digest
            != self.training_protocol_binding.input_view_digest
            or self.protocol_record.window_protocol_digest
            != self.training_protocol_binding.window_protocol_digest
            or self.protocol_record.training_protocol_digest
            != self.training_protocol_binding.training_protocol_digest
        ):
            raise EncoderProtocolError(
                "adapter protocol/access/view bindings are inconsistent"
            )
        for name in (
            "training_contract_digest",
            "abstract_fold_digest",
            "training_dataset_digest",
            "validation_dataset_digest",
            "semantic_output_protocol_digest",
            "runtime_digest",
        ):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "abstract_fold_id",
            checked_safe_id(self.abstract_fold_id, "abstract_fold_id"),
        )
        object.__setattr__(
            self,
            "execution_mode",
            checked_safe_id(self.execution_mode, "execution_mode"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EncoderProtocolError("adapter fit seed must be a non-negative integer")

    @property
    def adapter_contract_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-adapter-training-contract.v0",
                "protocol_record_digest": self.protocol_record.protocol_record_digest,
                "access_card_digest": self.access_card.access_card_digest,
                "training_protocol_binding_digest": (
                    self.training_protocol_binding.training_protocol_digest
                ),
                "training_contract_digest": self.training_contract_digest,
                "abstract_fold_id": self.abstract_fold_id,
                "abstract_fold_digest": self.abstract_fold_digest,
                "training_dataset_digest": self.training_dataset_digest,
                "validation_dataset_digest": self.validation_dataset_digest,
                "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
                "runtime_digest": self.runtime_digest,
                "execution_mode": self.execution_mode,
                "seed": self.seed,
            }
        )


def adapter_training_manifest_digest(
    train: AdapterFitData,
    validation: AdapterFitData,
    contract: AdapterTrainingContract,
) -> str:
    validate_adapter_fit_invocation(train, validation, contract)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-adapter-training-manifest.v0",
            "adapter_contract_digest": contract.adapter_contract_digest,
            "train_adapter_data_digest": train.adapter_data_digest,
            "validation_adapter_data_digest": validation.adapter_data_digest,
        }
    )


def validate_adapter_fit_invocation(
    train: AdapterFitData,
    validation: AdapterFitData,
    contract: AdapterTrainingContract,
) -> None:
    if not isinstance(train, AdapterFitData) or train.logical_role != "source_encoder_train":
        raise EncoderProtocolError("adapter fit requires sanitized training data")
    if (
        not isinstance(validation, AdapterFitData)
        or validation.logical_role != "source_encoder_validation"
    ):
        raise EncoderProtocolError("adapter fit requires sanitized validation data")
    if not isinstance(contract, AdapterTrainingContract):
        raise EncoderProtocolError("adapter fit requires AdapterTrainingContract")
    binding = contract.training_protocol_binding
    for name, data, expected_digest in (
        ("train", train, contract.training_dataset_digest),
        ("validation", validation, contract.validation_dataset_digest),
    ):
        if data.dataset_digest != expected_digest:
            raise EncoderProtocolError(f"adapter {name} dataset binding mismatch")
        if data.inputs.input_view_digest != binding.input_view_digest:
            raise EncoderProtocolError(f"adapter {name} uses an unknown input view")
        if data.inputs.window_protocol_digest != binding.window_protocol_digest:
            raise EncoderProtocolError(f"adapter {name} window protocol mismatch")
        if data.inputs.channel_allowlist != binding.channel_allowlist:
            raise EncoderProtocolError(f"adapter {name} channel view mismatch")
        validate_encoder_input_access(data.inputs, contract.access_card)
        if data.supervision is not None:
            unknown = set(data.supervision.labels) - set(_SUPERVISION_CAPABILITIES)
            if unknown:
                raise EncoderProtocolError(
                    f"unknown supervision channels cannot be authorized: {sorted(unknown)}"
                )
            contract.access_card.assert_can_read(
                *(
                    _SUPERVISION_CAPABILITIES[label]
                    for label in data.supervision.labels
                )
            )


@dataclass(frozen=True)
class SemanticSampleBatch:
    values: np.ndarray
    valid_mask: np.ndarray
    window_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        raw_mask = np.asarray(self.valid_mask)
        if raw_mask.dtype.kind != "b":
            raise EncoderProtocolError("semantic valid_mask must be boolean")
        mask = np.array(raw_mask, dtype=np.bool_, copy=True)
        if values.dtype != np.float64 or values.ndim != 2 or any(
            size <= 0 for size in values.shape
        ):
            raise EncoderProtocolError("semantic values must be a non-empty float64 matrix")
        if not np.all(np.isfinite(values)):
            raise EncoderProtocolError("semantic values must be finite")
        if mask.shape != (values.shape[0],):
            raise EncoderProtocolError("semantic valid_mask must have shape [num_windows]")
        if not np.any(mask):
            raise EncoderProtocolError(
                "semantic valid_mask must retain at least one valid window"
            )
        if len(self.window_ids) != values.shape[0] or len(set(self.window_ids)) != len(
            self.window_ids
        ):
            raise EncoderProtocolError("semantic window_ids must be unique and row-aligned")
        frozen_values = np.array(values, copy=True)
        frozen_mask = np.array(mask, copy=True)
        frozen_values.setflags(write=False)
        frozen_mask.setflags(write=False)
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(self, "valid_mask", frozen_mask)
        object.__setattr__(self, "window_ids", tuple(self.window_ids))

    @property
    def semantic_batch_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-semantic-sample-batch.v0",
                "arrays_digest": sha256_ndarrays(
                    {"values": self.values, "valid_mask": self.valid_mask}
                ),
                "window_ids": list(self.window_ids),
            }
        )


@runtime_checkable
class SemanticEncoderProtocol(Protocol):
    encoder_family: str
    protocol_record_digest: str
    access_card_digest: str
    semantic_output_protocol_digest: str

    def fit(
        self,
        train: AdapterFitData,
        validation: AdapterFitData,
        contract: AdapterTrainingContract,
    ) -> EncoderFitResult: ...

    def load_frozen(self, fit: EncoderFitResult) -> None: ...

    def encode_windows(
        self,
        inputs: SanitizedEncoderInputBatch,
        *,
        inference_contract: EncoderInferenceContract,
    ) -> SemanticSampleBatch: ...


_SUPERVISION_CAPABILITIES = {
    "task_label": SOURCE_TASK_CATEGORICAL_LABELS,
    "anchor_label": SOURCE_ANCHOR_CATEGORICAL_LABELS,
    "axis_label": SOURCE_AXIS_CATEGORICAL_LABELS,
    "numeric_factor": SOURCE_NUMERIC_FACTOR_PARAMETERS,
    "physical_parameter": SOURCE_NUMERIC_FACTOR_PARAMETERS,
    "probe_style_label": PROBE_STYLE_LABELS,
}


def validate_supervision_access(
    data: EncoderTrainingData | EncoderValidationData,
    access_card: EncoderAccessCard,
) -> None:
    if data.supervision is None:
        return
    unknown = set(data.supervision.labels) - set(_SUPERVISION_CAPABILITIES)
    if unknown:
        raise EncoderProtocolError(
            f"unknown supervision channels cannot be authorized: {sorted(unknown)}"
        )
    access_card.assert_can_read(
        *(_SUPERVISION_CAPABILITIES[name] for name in data.supervision.labels)
    )


def validate_encoder_input_access(
    inputs: SanitizedEncoderInputBatch,
    access_card: EncoderAccessCard,
) -> None:
    if not isinstance(inputs, SanitizedEncoderInputBatch):
        raise EncoderProtocolError("adapter input must be SanitizedEncoderInputBatch")
    if inputs.access_card_digest != access_card.access_card_digest:
        raise EncoderProtocolError("sanitized input/access-card digest mismatch")
    access_card.assert_can_read(
        SOURCE_RAW_TRANSITIONS, SOURCE_EPISODE_PROBE_BOUNDARIES
    )
    access_card.assert_can_read(
        *(_INPUT_CHANNEL_CAPABILITY[name] for name in inputs.channel_allowlist)
    )


def validate_training_data_contract(
    train: EncoderTrainingData,
    validation: EncoderValidationData,
    contract: EncoderTrainingContract,
) -> None:
    if not isinstance(train, EncoderTrainingData) or not isinstance(
        validation, EncoderValidationData
    ):
        raise EncoderProtocolError("fit requires typed train/validation data")
    if not isinstance(contract, EncoderTrainingContract):
        raise EncoderProtocolError("fit requires EncoderTrainingContract")
    expected_manifest = contract.role_manifest.manifest_digest
    expected_fold = contract.loto_fold.fold_record_digest
    for name, data in (("train", train), ("validation", validation)):
        if data.role_manifest.manifest_digest != expected_manifest:
            raise EncoderProtocolError(f"{name} role manifest differs from contract")
        if data.loto_fold.fold_record_digest != expected_fold:
            raise EncoderProtocolError(f"{name} LOTO fold differs from contract")
        if data.target_query_role != contract.target_query_role:
            raise EncoderProtocolError(f"{name} query role differs from contract")
        if data.inputs.input_view_digest != contract.protocol_record.input_view_digest:
            raise EncoderProtocolError(f"{name} input-view digest mismatch")
        if (
            data.inputs.window_protocol_digest
            != contract.protocol_record.window_protocol_digest
        ):
            raise EncoderProtocolError(f"{name} window protocol digest mismatch")
        if (
            data.inputs.channel_allowlist
            != contract.training_protocol_binding.channel_allowlist
        ):
            raise EncoderProtocolError(f"{name} channel view mismatch")
        validate_encoder_input_access(data.inputs, contract.access_card)
        validate_supervision_access(data, contract.access_card)

    if train.role_record.dataset_digest not in set(
        contract.loto_fold.train_dataset_digests
    ):
        raise EncoderProtocolError("training dataset is absent from the LOTO fold")
    if validation.role_record.dataset_digest not in set(
        contract.loto_fold.validation_dataset_digests
    ):
        raise EncoderProtocolError("validation dataset is absent from the LOTO fold")


def project_adapter_fit(
    train: EncoderTrainingData,
    validation: EncoderValidationData,
    contract: EncoderTrainingContract,
) -> tuple[AdapterFitData, AdapterFitData, AdapterTrainingContract]:
    """Validate private framework records, then project an adapter-safe fit call."""

    validate_training_data_contract(train, validation, contract)
    abstract_fold_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-abstract-loto-fold-binding.v0",
            "training_contract_digest": contract.contract_digest,
            "role_manifest_digest": contract.role_manifest.manifest_digest,
            "loto_fold_record_digest": contract.loto_fold.fold_record_digest,
            "train_dataset_digest": train.dataset_digest,
            "validation_dataset_digest": validation.dataset_digest,
        }
    )
    adapter_train = AdapterFitData(
        logical_role="source_encoder_train",
        inputs=train.inputs,
        supervision=train.supervision,
        dataset_digest=train.dataset_digest,
    )
    adapter_validation = AdapterFitData(
        logical_role="source_encoder_validation",
        inputs=validation.inputs,
        supervision=validation.supervision,
        dataset_digest=validation.dataset_digest,
    )
    adapter_contract = AdapterTrainingContract(
        protocol_record=contract.protocol_record,
        access_card=contract.access_card,
        training_protocol_binding=contract.training_protocol_binding,
        training_contract_digest=contract.contract_digest,
        abstract_fold_id=f"abstract-fold-{abstract_fold_digest[:24]}",
        abstract_fold_digest=abstract_fold_digest,
        training_dataset_digest=train.dataset_digest,
        validation_dataset_digest=validation.dataset_digest,
        semantic_output_protocol_digest=contract.semantic_output_protocol_digest,
        runtime_digest=contract.runtime_digest,
        execution_mode=contract.execution_mode,
        seed=contract.seed,
    )
    validate_adapter_fit_invocation(
        adapter_train, adapter_validation, adapter_contract
    )
    return adapter_train, adapter_validation, adapter_contract


def validate_fit_result_bindings(
    fit: EncoderFitResult,
    train: AdapterFitData,
    validation: AdapterFitData,
    contract: AdapterTrainingContract,
) -> None:
    """Independently verify every adapter-controlled fit-result binding."""

    if not isinstance(fit, EncoderFitResult):
        raise EncoderProtocolError("adapter fit must return EncoderFitResult")
    validate_adapter_fit_invocation(train, validation, contract)
    expected = {
        "encoder_id": contract.protocol_record.encoder_id,
        "training_manifest_digest": adapter_training_manifest_digest(
            train, validation, contract
        ),
        "protocol_record_digest": contract.protocol_record.protocol_record_digest,
        "training_contract_digest": contract.training_contract_digest,
        "access_card_digest": contract.access_card.access_card_digest,
        "input_view_digest": contract.training_protocol_binding.input_view_digest,
        "semantic_output_protocol_digest": contract.semantic_output_protocol_digest,
        "runtime_digest": contract.runtime_digest,
        "fold_id": contract.abstract_fold_id,
        "seed": contract.seed,
        "latent_dim": contract.protocol_record.latent_dim,
    }
    drift = {
        name: {"expected": expected_value, "observed": getattr(fit, name)}
        for name, expected_value in expected.items()
        if getattr(fit, name) != expected_value
    }
    if drift:
        raise EncoderProtocolError(
            f"adapter fit result bindings drifted: {sorted(drift)}"
        )
    if fit.training_cost.trial_count > contract.access_card.max_hyperparameter_trials:
        raise EncoderProtocolError(
            "fit trial_count exceeds access-card max_hyperparameter_trials"
        )


def validate_semantic_output(
    output: SemanticSampleBatch,
    inputs: SanitizedEncoderInputBatch,
    *,
    latent_dim: int,
) -> None:
    if not isinstance(output, SemanticSampleBatch):
        raise EncoderProtocolError("encoder output must be SemanticSampleBatch")
    if output.values.shape != (len(inputs.window_ids), latent_dim):
        raise EncoderProtocolError("encoder output shape does not match contract")
    if output.window_ids != inputs.window_ids:
        raise EncoderProtocolError("encoder output reordered or replaced window IDs")


__all__ = [
    "AdapterFitData",
    "AdapterTrainingContract",
    "CostRecord",
    "ENCODER_INPUT_CHANNELS",
    "EncoderFitResult",
    "EncoderInferenceContract",
    "EncoderProtocolError",
    "EncoderSupervisionBatch",
    "EncoderTrainingContract",
    "EncoderTrainingData",
    "EncoderTrainingProtocolBinding",
    "EncoderValidationData",
    "SanitizedEncoderInputBatch",
    "SemanticEncoderProtocol",
    "SemanticSampleBatch",
    "adapter_training_manifest_digest",
    "encoder_dataset_digest",
    "project_adapter_fit",
    "sanitize_encoder_inputs",
    "validate_adapter_fit_invocation",
    "validate_encoder_input_access",
    "validate_fit_result_bindings",
    "validate_semantic_output",
    "validate_supervision_access",
    "validate_training_data_contract",
]
