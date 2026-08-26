"""Formal representation and pair/bank controls for the v0.3 Signal Atlas.

These controls deliberately do not alter the frozen 13-input-view registry.
The historical random-tanh entry is typed as a representation control;
reward-free shuffled-next is a condition based on the reward-free view; and
schema-collision/exact-repeat are pair-level contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from .signal_metrics import SignalDistanceRow, SignalMetricRecord
from .transition_views import (
    VIEW_REGISTRY,
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


HISTORICAL_RANDOM_TANH_ID = "R_HIST_RANDOM_TANH"
MATCHED_RANDOM_MLP_ID = "R3_MATCHED_RANDOM_MLP"
HISTORICAL_RANDOM_TANH_CELL_ID = "CELL_HIST_RANDOM_TANH"
RF_SHUFFLED_NEXT_CONTROL_ID = "C_RF_SHUFFLED_NEXT"
SCHEMA_COLLISION_CONTROL_ID = "C_SCHEMA_COLLISION"
EXACT_REPEAT_CONTROL_ID = "C_EXACT_REPEAT"

HISTORICAL_RANDOM_TANH_SCHEMA = (
    "policy-learnware.v03-historical-random-tanh.v0"
)
HISTORICAL_RANDOM_TANH_RESULT_SCHEMA = (
    "policy-learnware.v03-historical-random-tanh-result.v0"
)
RF_SHUFFLED_NEXT_SCHEMA = "policy-learnware.v03-rf-shuffled-next.v0"
RF_SHUFFLED_NEXT_RESULT_SCHEMA = (
    "policy-learnware.v03-rf-shuffled-next-result.v0"
)
BANK_CONTROL_REFERENCE_SCHEMA = "policy-learnware.v03-bank-control-reference.v1"
SCHEMA_COLLISION_PAIR_SCHEMA = "policy-learnware.v03-schema-collision-pair.v0"
EXACT_REPEAT_PAIR_SCHEMA = "policy-learnware.v03-exact-repeat-pair.v0"
PAIR_CONTROL_MEMBERSHIP_EVIDENCE_SCHEMA = (
    "policy-learnware.v03-pair-control-membership-evidence.v0"
)
PAIR_CONTROL_RESULT_SCHEMA = "policy-learnware.v03-pair-control-result.v0"
PAIR_CONTROL_EVALUATION_SCHEMA = (
    "policy-learnware.v03-pair-control-evaluation.v0"
)
PAIR_CONTROL_PLAN_SCHEMA = "policy-learnware.v03-pair-control-plan.v0"
PUBLIC_PAIR_CONTROL_RESULT_SCHEMA = (
    "policy-learnware.v03-public-pair-control-result.v0"
)
EXACT_REPEAT_DISTANCE_RESULT_SCHEMA = (
    "policy-learnware.v03-exact-repeat-distance-result.v0"
)
PUBLIC_EXACT_REPEAT_DISTANCE_SCHEMA = (
    "policy-learnware.v03-public-exact-repeat-distance.v0"
)
EXACT_REPEAT_NOISE_RATIO_SCHEMA = (
    "policy-learnware.v03-exact-repeat-noise-ratio.v0"
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class SignalControlError(ValueError):
    """A formal signal control is ambiguous, mutable, or invalid."""


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise SignalControlError(f"{where} must be a canonical safe ID")
    return value


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise SignalControlError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SignalControlError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SignalControlError(f"{where} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise SignalControlError(f"{where} must be a positive integer")
    return result


def _seed(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SignalControlError(f"{where} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise SignalControlError(f"{where} must be a non-negative integer")
    return result


def _readonly_matrix(
    value: Any, *, where: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise SignalControlError(f"{where} must be a finite array of shape {shape}")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _random_tanh_parameters(
    *, seed: int, input_dim: int, output_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(max(1, input_dim))
    matrix = rng.normal(0.0, scale, size=(input_dim, output_dim)).astype(np.float32)
    bias = rng.normal(0.0, scale, size=(output_dim,)).astype(np.float32)
    return matrix, bias


@dataclass(frozen=True)
class HistoricalRandomTanhResult:
    representation_id: str
    representation_protocol_digest: str
    checkpoint_digest: str
    base_view_digest: str
    source_bank_digest: str
    values: np.ndarray
    result_digest: str | None = None
    schema: str = HISTORICAL_RANDOM_TANH_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_RANDOM_TANH_RESULT_SCHEMA:
            raise SignalControlError("unsupported HistoricalRandomTanhResult schema")
        if self.representation_id != HISTORICAL_RANDOM_TANH_ID:
            raise SignalControlError("historical random result has the wrong identity")
        for name in (
            "representation_protocol_digest",
            "checkpoint_digest",
            "base_view_digest",
            "source_bank_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        raw = np.asarray(self.values)
        if (
            raw.dtype != np.float32
            or raw.ndim != 2
            or any(size <= 0 for size in raw.shape)
            or not np.all(np.isfinite(raw))
        ):
            raise SignalControlError(
                "historical random values must be a non-empty finite float32 matrix"
            )
        values = np.ascontiguousarray(raw).copy()
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        expected = sha256_json(self._payload_without_digest())
        if self.result_digest is None:
            object.__setattr__(self, "result_digest", expected)
        elif _digest(self.result_digest, "result_digest") != expected:
            raise SignalControlError("historical random result digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_id": self.representation_id,
            "representation_protocol_digest": self.representation_protocol_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "base_view_digest": self.base_view_digest,
            "source_bank_digest": self.source_bank_digest,
            "values_digest": sha256_ndarrays({"values": self.values}),
            "row_count": int(self.values.shape[0]),
            "output_dim": int(self.values.shape[1]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "result_digest": self.result_digest}


@dataclass(frozen=True)
class HistoricalRandomTanhSpec:
    seed: int
    input_dim: int
    output_dim: int
    matrix: np.ndarray
    bias: np.ndarray
    representation_id: str = HISTORICAL_RANDOM_TANH_ID
    cell_id: str = HISTORICAL_RANDOM_TANH_CELL_ID
    base_view_id: str = V_FULL_LEGACY
    activation: str = "tanh"
    initialization: str = "numpy-pcg64-normal-fan-in-v0"
    representation_protocol_digest: str | None = None
    checkpoint_digest: str | None = None
    schema: str = HISTORICAL_RANDOM_TANH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HISTORICAL_RANDOM_TANH_SCHEMA:
            raise SignalControlError("unsupported HistoricalRandomTanhSpec schema")
        if self.representation_id != HISTORICAL_RANDOM_TANH_ID:
            raise SignalControlError("historical control identity is frozen")
        if self.representation_id == MATCHED_RANDOM_MLP_ID:
            raise SignalControlError("historical random tanh cannot be identified as R3")
        if self.cell_id != HISTORICAL_RANDOM_TANH_CELL_ID:
            raise SignalControlError("historical random cell identity is frozen")
        if self.base_view_id != V_FULL_LEGACY:
            raise SignalControlError("historical random tanh must consume FULL legacy")
        if self.activation != "tanh" or self.initialization != "numpy-pcg64-normal-fan-in-v0":
            raise SignalControlError("historical random architecture is frozen")
        seed = _seed(self.seed, "seed")
        input_dim = _positive_int(self.input_dim, "input_dim")
        output_dim = _positive_int(self.output_dim, "output_dim")
        matrix = _readonly_matrix(
            self.matrix, where="matrix", shape=(input_dim, output_dim)
        )
        bias = _readonly_matrix(self.bias, where="bias", shape=(output_dim,))
        expected_matrix, expected_bias = _random_tanh_parameters(
            seed=seed, input_dim=input_dim, output_dim=output_dim
        )
        if not np.array_equal(matrix, expected_matrix) or not np.array_equal(
            bias, expected_bias
        ):
            raise SignalControlError("matrix/bias do not match the frozen seed recipe")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "output_dim", output_dim)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)
        expected_protocol = sha256_json(self._protocol_payload())
        if self.representation_protocol_digest is None:
            object.__setattr__(
                self, "representation_protocol_digest", expected_protocol
            )
        elif (
            _digest(
                self.representation_protocol_digest,
                "representation_protocol_digest",
            )
            != expected_protocol
        ):
            raise SignalControlError("representation protocol digest mismatch")
        expected_checkpoint = sha256_json(
            {
                "schema": "policy-learnware.v03-historical-random-tanh-checkpoint.v0",
                "representation_protocol_digest": expected_protocol,
                "seed": seed,
                "parameter_digest": self.parameter_digest,
            }
        )
        if self.checkpoint_digest is None:
            object.__setattr__(self, "checkpoint_digest", expected_checkpoint)
        elif _digest(self.checkpoint_digest, "checkpoint_digest") != expected_checkpoint:
            raise SignalControlError("historical random checkpoint digest mismatch")

    @classmethod
    def create(
        cls, *, seed: int, input_dim: int, output_dim: int
    ) -> "HistoricalRandomTanhSpec":
        resolved_seed = _seed(seed, "seed")
        resolved_input = _positive_int(input_dim, "input_dim")
        resolved_output = _positive_int(output_dim, "output_dim")
        matrix, bias = _random_tanh_parameters(
            seed=resolved_seed,
            input_dim=resolved_input,
            output_dim=resolved_output,
        )
        return cls(
            seed=resolved_seed,
            input_dim=resolved_input,
            output_dim=resolved_output,
            matrix=matrix,
            bias=bias,
        )

    @property
    def parameter_digest(self) -> str:
        return sha256_ndarrays({"matrix": self.matrix, "bias": self.bias})

    @property
    def is_matched_random_mlp(self) -> bool:
        return False

    def _protocol_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_id": self.representation_id,
            "cell_id": self.cell_id,
            "base_view_id": self.base_view_id,
            "base_view_spec_digest": VIEW_REGISTRY[V_FULL_LEGACY].digest,
            "architecture": "one-random-affine-then-tanh",
            "activation": self.activation,
            "initialization": self.initialization,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._protocol_payload(),
            "seed": self.seed,
            "parameter_digest": self.parameter_digest,
            "representation_protocol_digest": self.representation_protocol_digest,
            "checkpoint_digest": self.checkpoint_digest,
        }

    def apply(self, bank: TransitionBank) -> HistoricalRandomTanhResult:
        if not isinstance(bank, TransitionBank):
            raise SignalControlError("historical random control requires TransitionBank")
        full = apply_transition_view(bank, V_FULL_LEGACY)
        values = full.feature_matrix
        if values.shape[1] != self.input_dim:
            raise SignalControlError(
                "FULL feature width differs from historical random input_dim"
            )
        historical_view = apply_transition_view(
            bank,
            V_RANDOM_ENCODER,
            shuffle_seed=self.seed,
            random_output_dim=self.output_dim,
        )
        if historical_view.random_projection_digest != self.parameter_digest:
            raise SignalControlError(
                "historical view parameters differ from the canonical control spec"
            )
        encoded = np.asarray(
            historical_view.channels["random_embedding"], dtype=np.float32
        )
        return HistoricalRandomTanhResult(
            representation_id=self.representation_id,
            representation_protocol_digest=str(
                self.representation_protocol_digest
            ),
            checkpoint_digest=str(self.checkpoint_digest),
            base_view_digest=full.view_digest,
            source_bank_digest=bank.canonical_bank_digest,
            values=encoded,
        )


def _nontrivial_permutation(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise SignalControlError("RF shuffled-next requires at least two rows")
    permutation = np.random.default_rng(seed).permutation(size).astype(np.int64)
    identity = np.arange(size, dtype=np.int64)
    if np.array_equal(permutation, identity):
        permutation = np.roll(identity, 1)
    permutation.setflags(write=False)
    return permutation


def _pairing_destroying_permutation(values: np.ndarray, seed: int) -> np.ndarray:
    """Return a permutation that changes the observed row pairing, if possible."""

    rows = np.asarray(values)
    permutation = _nontrivial_permutation(rows.shape[0], seed)
    if not np.array_equal(rows[permutation], rows):
        return permutation
    # A non-identity index permutation can be observationally null when it only
    # exchanges duplicate next states.  Construct a deterministic swap between
    # two genuinely different rows; if none exist, the control is inapplicable.
    for left in range(rows.shape[0] - 1):
        different = np.flatnonzero(
            np.any(rows[left + 1 :] != rows[left], axis=1)
        )
        if different.size:
            right = left + 1 + int(different[0])
            result = np.arange(rows.shape[0], dtype=np.int64)
            result[left], result[right] = result[right], result[left]
            result.setflags(write=False)
            return result
    raise SignalControlError(
        "RF shuffled-next cannot destroy pairing because all next observations repeat"
    )


def _row_multiset_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    row_digests = [
        sha256_ndarrays({"row": np.ascontiguousarray(array[index : index + 1])})
        for index in range(array.shape[0])
    ]
    return sha256_json(
        {
            "schema": "policy-learnware.v03-row-multiset.v0",
            "dtype": array.dtype.str,
            "row_shape": list(array.shape[1:]),
            "row_digests": sorted(row_digests),
        }
    )


@dataclass(frozen=True)
class MarginalPreservationAudit:
    observation_exact: bool
    action_exact: bool
    next_observation_multiset_exact: bool
    pairing_destroyed: bool
    reward_absent: bool
    mask_absent: bool
    base_observation_marginal_digest: str
    control_observation_marginal_digest: str
    base_action_marginal_digest: str
    control_action_marginal_digest: str
    base_next_marginal_digest: str
    control_next_marginal_digest: str

    def __post_init__(self) -> None:
        for name in (
            "observation_exact",
            "action_exact",
            "next_observation_multiset_exact",
            "pairing_destroyed",
            "reward_absent",
            "mask_absent",
        ):
            if type(getattr(self, name)) is not bool:
                raise SignalControlError(f"{name} must be boolean")
        for name in (
            "base_observation_marginal_digest",
            "control_observation_marginal_digest",
            "base_action_marginal_digest",
            "control_action_marginal_digest",
            "base_next_marginal_digest",
            "control_next_marginal_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not self.passed:
            raise SignalControlError("RF shuffled-next marginal audit did not pass")

    @property
    def passed(self) -> bool:
        return all(
            (
                self.observation_exact,
                self.action_exact,
                self.next_observation_multiset_exact,
                self.pairing_destroyed,
                self.reward_absent,
                self.mask_absent,
            )
        )

    @property
    def audit_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-rf-shuffled-marginal-audit.v0",
                **self.__dict__,
            }
        )


@dataclass(frozen=True)
class RewardFreeShuffledNextResult:
    control_id: str
    base_view_id: str
    transform_digest: str
    base_view_digest: str
    source_bank_digest: str
    channels: Mapping[str, np.ndarray]
    next_source_indices: np.ndarray
    marginal_audit: MarginalPreservationAudit
    dataset_digest: str | None = None
    schema: str = RF_SHUFFLED_NEXT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RF_SHUFFLED_NEXT_RESULT_SCHEMA:
            raise SignalControlError("unsupported RF shuffled-next result schema")
        if self.control_id != RF_SHUFFLED_NEXT_CONTROL_ID:
            raise SignalControlError("RF shuffled-next result has wrong identity")
        if self.base_view_id != V_REWARD_FREE_TRANSITION:
            raise SignalControlError("RF shuffled-next must use reward-free base")
        for name in ("transform_digest", "base_view_digest", "source_bank_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if set(self.channels) != {"observation", "action", "next_observation"}:
            raise SignalControlError(
                "RF shuffled-next exposes forbidden reward/mask channels"
            )
        frozen: dict[str, np.ndarray] = {}
        rows: int | None = None
        for name in ("observation", "action", "next_observation"):
            value = np.asarray(self.channels[name], dtype=np.float32)
            if value.ndim != 2 or value.shape[0] <= 0 or not np.all(np.isfinite(value)):
                raise SignalControlError(f"RF channel {name} must be a finite matrix")
            if rows is None:
                rows = int(value.shape[0])
            elif value.shape[0] != rows:
                raise SignalControlError("RF control channel row counts disagree")
            copied = np.ascontiguousarray(value).copy()
            copied.setflags(write=False)
            frozen[name] = copied
        raw_indices = np.asarray(self.next_source_indices)
        if raw_indices.dtype.kind not in "iu" or raw_indices.shape != (rows,):
            raise SignalControlError("next_source_indices must align with rows")
        indices = np.asarray(raw_indices, dtype=np.int64).copy()
        if set(indices.tolist()) != set(range(int(rows))):
            raise SignalControlError("next_source_indices must be a row permutation")
        if np.array_equal(indices, np.arange(int(rows), dtype=np.int64)):
            raise SignalControlError("next_source_indices must be nontrivial")
        indices.setflags(write=False)
        if not isinstance(self.marginal_audit, MarginalPreservationAudit):
            raise SignalControlError("RF result requires typed marginal audit")
        observed_control_marginals = {
            "observation": _row_multiset_digest(frozen["observation"]),
            "action": _row_multiset_digest(frozen["action"]),
            "next_observation": _row_multiset_digest(
                frozen["next_observation"]
            ),
        }
        audit = self.marginal_audit
        if (
            audit.control_observation_marginal_digest
            != observed_control_marginals["observation"]
            or audit.control_action_marginal_digest
            != observed_control_marginals["action"]
            or audit.control_next_marginal_digest
            != observed_control_marginals["next_observation"]
            or audit.base_observation_marginal_digest
            != audit.control_observation_marginal_digest
            or audit.base_action_marginal_digest
            != audit.control_action_marginal_digest
            or audit.base_next_marginal_digest
            != audit.control_next_marginal_digest
        ):
            raise SignalControlError(
                "RF shuffled-next audit digests disagree with result marginals"
            )
        object.__setattr__(self, "channels", MappingProxyType(frozen))
        object.__setattr__(self, "next_source_indices", indices)
        expected = sha256_json(self._payload_without_digest())
        if self.dataset_digest is None:
            object.__setattr__(self, "dataset_digest", expected)
        elif _digest(self.dataset_digest, "dataset_digest") != expected:
            raise SignalControlError("RF shuffled-next dataset digest mismatch")

    @property
    def feature_matrix(self) -> np.ndarray:
        result = np.concatenate(
            [
                self.channels["observation"],
                self.channels["action"],
                self.channels["next_observation"],
            ],
            axis=1,
        )
        result.setflags(write=False)
        return result

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.control_id,
            "base_view_id": self.base_view_id,
            "transform_digest": self.transform_digest,
            "base_view_digest": self.base_view_digest,
            "source_bank_digest": self.source_bank_digest,
            "arrays_digest": sha256_ndarrays(
                {**self.channels, "next_source_indices": self.next_source_indices}
            ),
            "marginal_audit_digest": self.marginal_audit.audit_digest,
        }


@dataclass(frozen=True)
class RewardFreeShuffledNextSpec:
    seed: int
    control_id: str = RF_SHUFFLED_NEXT_CONTROL_ID
    base_view_id: str = V_REWARD_FREE_TRANSITION
    permutation_scope: str = "within-bank"
    transform_digest: str | None = None
    schema: str = RF_SHUFFLED_NEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RF_SHUFFLED_NEXT_SCHEMA:
            raise SignalControlError("unsupported RewardFreeShuffledNextSpec schema")
        if self.control_id != RF_SHUFFLED_NEXT_CONTROL_ID:
            raise SignalControlError("RF shuffled-next identity is frozen")
        if self.base_view_id != V_REWARD_FREE_TRANSITION:
            raise SignalControlError("RF shuffled-next base must be reward-free transition")
        if self.permutation_scope != "within-bank":
            raise SignalControlError("RF shuffled-next permutation scope is frozen")
        object.__setattr__(self, "seed", _seed(self.seed, "seed"))
        expected = sha256_json(self._payload_without_digest())
        if self.transform_digest is None:
            object.__setattr__(self, "transform_digest", expected)
        elif _digest(self.transform_digest, "transform_digest") != expected:
            raise SignalControlError("RF shuffled-next transform digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.control_id,
            "base_view_id": self.base_view_id,
            "base_view_spec_digest": VIEW_REGISTRY[
                V_REWARD_FREE_TRANSITION
            ].digest,
            "channel_allowlist": [
                "observation",
                "action",
                "next_observation",
            ],
            "permutation_scope": self.permutation_scope,
            "seed": self.seed,
            "forbidden_channels": [
                "reward",
                "terminated",
                "truncated",
                "observation_mask",
                "action_mask",
                "next_observation_mask",
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "transform_digest": self.transform_digest}

    def apply(self, bank: TransitionBank) -> RewardFreeShuffledNextResult:
        if not isinstance(bank, TransitionBank):
            raise SignalControlError("RF shuffled-next requires TransitionBank")
        base = apply_transition_view(bank, V_REWARD_FREE_TRANSITION)
        base_next = np.asarray(base.channels["next_observation"], dtype=np.float32)
        permutation = _pairing_destroying_permutation(base_next, self.seed)
        observation = np.asarray(base.channels["observation"], dtype=np.float32)
        action = np.asarray(base.channels["action"], dtype=np.float32)
        control_next = base_next[permutation]
        channels = {
            "observation": observation,
            "action": action,
            "next_observation": control_next,
        }
        observation_digest = _row_multiset_digest(observation)
        action_digest = _row_multiset_digest(action)
        base_next_digest = _row_multiset_digest(base_next)
        control_next_digest = _row_multiset_digest(control_next)
        audit = MarginalPreservationAudit(
            observation_exact=np.array_equal(observation, base.channels["observation"]),
            action_exact=np.array_equal(action, base.channels["action"]),
            next_observation_multiset_exact=(
                base_next_digest == control_next_digest
                and np.array_equal(control_next, base_next[permutation])
            ),
            pairing_destroyed=not np.array_equal(control_next, base_next),
            reward_absent="reward" not in channels,
            mask_absent=not any("mask" in name for name in channels),
            base_observation_marginal_digest=observation_digest,
            control_observation_marginal_digest=_row_multiset_digest(
                channels["observation"]
            ),
            base_action_marginal_digest=action_digest,
            control_action_marginal_digest=_row_multiset_digest(channels["action"]),
            base_next_marginal_digest=base_next_digest,
            control_next_marginal_digest=control_next_digest,
        )
        return RewardFreeShuffledNextResult(
            control_id=self.control_id,
            base_view_id=self.base_view_id,
            transform_digest=str(self.transform_digest),
            base_view_digest=base.view_digest,
            source_bank_digest=bank.canonical_bank_digest,
            channels=channels,
            next_source_indices=permutation,
            marginal_audit=audit,
        )


@dataclass(frozen=True)
class BankControlReference:
    bank_id: str
    registered_task_id: str
    embodiment_id: str
    abi_contract_id: str
    goal_contract_id: str
    dynamics_context_id: str
    context_id: str
    observation_dim: int
    action_dim: int
    bank_digest: str
    measurement_protocol_digest: str
    probe_seed_digest: str
    schema: str = BANK_CONTROL_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BANK_CONTROL_REFERENCE_SCHEMA:
            raise SignalControlError("unsupported BankControlReference schema")
        for name in (
            "bank_id",
            "registered_task_id",
            "embodiment_id",
            "abi_contract_id",
            "goal_contract_id",
            "dynamics_context_id",
            "context_id",
        ):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        object.__setattr__(
            self,
            "observation_dim",
            _positive_int(self.observation_dim, "observation_dim"),
        )
        object.__setattr__(
            self, "action_dim", _positive_int(self.action_dim, "action_dim")
        )
        for name in (
            "bank_digest",
            "measurement_protocol_digest",
            "probe_seed_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def bank_reference_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bank_id": self.bank_id,
            "registered_task_id": self.registered_task_id,
            "embodiment_id": self.embodiment_id,
            "abi_contract_id": self.abi_contract_id,
            "goal_contract_id": self.goal_contract_id,
            "dynamics_context_id": self.dynamics_context_id,
            "context_id": self.context_id,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "bank_digest": self.bank_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "probe_seed_digest": self.probe_seed_digest,
        }


def _metric_contract(
    metric_ids: tuple[str, ...], statistical_identity: str
) -> tuple[tuple[str, ...], str, str]:
    metrics = tuple(sorted(_safe_id(item, "metric_id") for item in metric_ids))
    if not metrics or len(set(metrics)) != len(metrics):
        raise SignalControlError("metric_ids must be non-empty and unique")
    identity = _safe_id(statistical_identity, "statistical_identity")
    digest = sha256_json(
        {
            "schema": "policy-learnware.v03-control-metric-contract.v0",
            "metric_ids": list(metrics),
            "statistical_identity": identity,
        }
    )
    return metrics, identity, digest


@dataclass(frozen=True)
class SchemaCollisionPairContract:
    pair_id: str
    left: BankControlReference
    right: BankControlReference
    metric_ids: tuple[str, ...]
    statistical_identity: str
    preregistration_digest: str
    control_id: str = SCHEMA_COLLISION_CONTROL_ID
    pair_digest: str | None = None
    schema: str = SCHEMA_COLLISION_PAIR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_COLLISION_PAIR_SCHEMA:
            raise SignalControlError("unsupported SchemaCollisionPairContract schema")
        if self.control_id != SCHEMA_COLLISION_CONTROL_ID:
            raise SignalControlError("schema-collision control identity is frozen")
        object.__setattr__(self, "pair_id", _safe_id(self.pair_id, "pair_id"))
        if not isinstance(self.left, BankControlReference) or not isinstance(
            self.right, BankControlReference
        ):
            raise SignalControlError("schema-collision pair requires bank references")
        if self.left.bank_digest == self.right.bank_digest:
            raise SignalControlError("schema-collision pair requires distinct banks")
        if (
            self.left.observation_dim != self.right.observation_dim
            or self.left.action_dim != self.right.action_dim
        ):
            raise SignalControlError("schema-collision pair must share native dimensions")
        if (
            self.left.measurement_protocol_digest
            != self.right.measurement_protocol_digest
        ):
            raise SignalControlError(
                "schema-collision pair must share the measurement protocol"
            )
        if (
            self.left.registered_task_id == self.right.registered_task_id
            and self.left.goal_contract_id == self.right.goal_contract_id
        ):
            raise SignalControlError(
                "schema-collision pair must differ in registered task or goal"
            )
        metrics, identity, _ = _metric_contract(
            tuple(self.metric_ids), self.statistical_identity
        )
        object.__setattr__(self, "metric_ids", metrics)
        object.__setattr__(self, "statistical_identity", identity)
        object.__setattr__(
            self,
            "preregistration_digest",
            _digest(self.preregistration_digest, "preregistration_digest"),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.pair_digest is None:
            object.__setattr__(self, "pair_digest", expected)
        elif _digest(self.pair_digest, "pair_digest") != expected:
            raise SignalControlError("schema-collision pair digest mismatch")

    @property
    def metric_protocol_digest(self) -> str:
        return _metric_contract(self.metric_ids, self.statistical_identity)[2]

    @property
    def adds_input_view(self) -> bool:
        return False

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.control_id,
            "pair_id": self.pair_id,
            "left_bank_reference_digest": self.left.bank_reference_digest,
            "right_bank_reference_digest": self.right.bank_reference_digest,
            "metric_protocol_digest": self.metric_protocol_digest,
            "statistical_identity": self.statistical_identity,
            "preregistration_digest": self.preregistration_digest,
            "control_level": "PAIR_SUBSET",
            "adds_input_view": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "pair_digest": self.pair_digest}


@dataclass(frozen=True)
class ExactRepeatPairContract:
    pair_id: str
    left: BankControlReference
    right: BankControlReference
    metric_ids: tuple[str, ...]
    statistical_identity: str
    preregistration_digest: str
    control_id: str = EXACT_REPEAT_CONTROL_ID
    pair_digest: str | None = None
    schema: str = EXACT_REPEAT_PAIR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXACT_REPEAT_PAIR_SCHEMA:
            raise SignalControlError("unsupported ExactRepeatPairContract schema")
        if self.control_id != EXACT_REPEAT_CONTROL_ID:
            raise SignalControlError("exact-repeat control identity is frozen")
        object.__setattr__(self, "pair_id", _safe_id(self.pair_id, "pair_id"))
        if not isinstance(self.left, BankControlReference) or not isinstance(
            self.right, BankControlReference
        ):
            raise SignalControlError("exact-repeat pair requires bank references")
        same_context = (
            self.left.registered_task_id == self.right.registered_task_id
            and self.left.embodiment_id == self.right.embodiment_id
            and self.left.abi_contract_id == self.right.abi_contract_id
            and self.left.goal_contract_id == self.right.goal_contract_id
            and self.left.dynamics_context_id == self.right.dynamics_context_id
            and self.left.context_id == self.right.context_id
            and self.left.observation_dim == self.right.observation_dim
            and self.left.action_dim == self.right.action_dim
            and self.left.measurement_protocol_digest
            == self.right.measurement_protocol_digest
        )
        if not same_context:
            raise SignalControlError(
                "exact-repeat banks must share task, embodiment, ABI, goal, "
                "dynamics, collection context and measurement protocol"
            )
        if self.left.bank_digest == self.right.bank_digest:
            raise SignalControlError("exact-repeat requires independently collected banks")
        if self.left.probe_seed_digest == self.right.probe_seed_digest:
            raise SignalControlError("exact-repeat requires independent probe seeds")
        metrics, identity, _ = _metric_contract(
            tuple(self.metric_ids), self.statistical_identity
        )
        object.__setattr__(self, "metric_ids", metrics)
        object.__setattr__(self, "statistical_identity", identity)
        object.__setattr__(
            self,
            "preregistration_digest",
            _digest(self.preregistration_digest, "preregistration_digest"),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.pair_digest is None:
            object.__setattr__(self, "pair_digest", expected)
        elif _digest(self.pair_digest, "pair_digest") != expected:
            raise SignalControlError("exact-repeat pair digest mismatch")

    @property
    def metric_protocol_digest(self) -> str:
        return _metric_contract(self.metric_ids, self.statistical_identity)[2]

    @property
    def adds_input_view(self) -> bool:
        return False

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.control_id,
            "pair_id": self.pair_id,
            "left_bank_reference_digest": self.left.bank_reference_digest,
            "right_bank_reference_digest": self.right.bank_reference_digest,
            "metric_protocol_digest": self.metric_protocol_digest,
            "statistical_identity": self.statistical_identity,
            "preregistration_digest": self.preregistration_digest,
            "control_level": "BANK_REPEAT",
            "adds_input_view": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "pair_digest": self.pair_digest}


PairControlContract = SchemaCollisionPairContract | ExactRepeatPairContract


@dataclass(frozen=True)
class PairControlMembershipEvidence:
    """The exact measured feature banks and receipts used for one pair control.

    Pair contracts intentionally describe bank-level membership without adding
    an input view.  This evidence is the runtime join: it binds those frozen
    references to the concrete canonical receipts and feature-bank artifacts
    that produced a metric record.
    """

    pair_digest: str
    left_bank_reference_digest: str
    right_bank_reference_digest: str
    left_bank_id: str
    right_bank_id: str
    left_receipt_digest: str
    right_receipt_digest: str
    left_feature_bank_digest: str
    right_feature_bank_digest: str
    measurement_protocol_digest: str
    evidence_digest: str | None = None
    schema: str = PAIR_CONTROL_MEMBERSHIP_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_CONTROL_MEMBERSHIP_EVIDENCE_SCHEMA:
            raise SignalControlError(
                "unsupported PairControlMembershipEvidence schema"
            )
        for name in ("left_bank_id", "right_bank_id"):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        if self.left_bank_id == self.right_bank_id:
            raise SignalControlError("pair-control membership requires two bank IDs")
        for name in (
            "pair_digest",
            "left_bank_reference_digest",
            "right_bank_reference_digest",
            "left_receipt_digest",
            "right_receipt_digest",
            "left_feature_bank_digest",
            "right_feature_bank_digest",
            "measurement_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.left_receipt_digest == self.right_receipt_digest:
            raise SignalControlError(
                "pair-control membership requires distinct canonical receipts"
            )
        if self.left_feature_bank_digest == self.right_feature_bank_digest:
            raise SignalControlError(
                "pair-control membership requires distinct feature banks"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.evidence_digest is None:
            object.__setattr__(self, "evidence_digest", expected)
        elif _digest(self.evidence_digest, "evidence_digest") != expected:
            raise SignalControlError("pair-control membership evidence digest mismatch")

    @classmethod
    def create(
        cls,
        contract: PairControlContract,
        *,
        left_receipt_digest: str,
        right_receipt_digest: str,
        left_feature_bank_digest: str,
        right_feature_bank_digest: str,
    ) -> "PairControlMembershipEvidence":
        if not isinstance(
            contract, (SchemaCollisionPairContract, ExactRepeatPairContract)
        ):
            raise SignalControlError(
                "pair-control membership requires a typed pair contract"
            )
        return cls(
            pair_digest=str(contract.pair_digest),
            left_bank_reference_digest=contract.left.bank_reference_digest,
            right_bank_reference_digest=contract.right.bank_reference_digest,
            left_bank_id=contract.left.bank_id,
            right_bank_id=contract.right.bank_id,
            left_receipt_digest=left_receipt_digest,
            right_receipt_digest=right_receipt_digest,
            left_feature_bank_digest=left_feature_bank_digest,
            right_feature_bank_digest=right_feature_bank_digest,
            measurement_protocol_digest=(
                contract.left.measurement_protocol_digest
            ),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pair_digest": self.pair_digest,
            "left_bank_reference_digest": self.left_bank_reference_digest,
            "right_bank_reference_digest": self.right_bank_reference_digest,
            "left_bank_id": self.left_bank_id,
            "right_bank_id": self.right_bank_id,
            "left_receipt_digest": self.left_receipt_digest,
            "right_receipt_digest": self.right_receipt_digest,
            "left_feature_bank_digest": self.left_feature_bank_digest,
            "right_feature_bank_digest": self.right_feature_bank_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evidence_digest": self.evidence_digest}


@dataclass(frozen=True)
class PairControlResult:
    """Digest-stable metric result for a preregistered pair/bank control."""

    control_id: str
    pair_id: str
    pair_digest: str
    preregistration_digest: str
    statistical_identity: str
    metric_protocol_digest: str
    measurement_protocol_digest: str
    membership_evidence_digest: str
    source_metric_record_digest: str
    pair_metric_record_digest: str
    pair_filter_digest: str
    metric_record_cell_id: str
    metric_ids: tuple[str, ...]
    metric_values: Mapping[str, float]
    pair_query_count: int
    pair_source_count: int
    result_digest: str | None = None
    schema: str = PAIR_CONTROL_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_CONTROL_RESULT_SCHEMA:
            raise SignalControlError("unsupported PairControlResult schema")
        if self.control_id not in {
            SCHEMA_COLLISION_CONTROL_ID,
            EXACT_REPEAT_CONTROL_ID,
        }:
            raise SignalControlError("pair-control result has an unknown control ID")
        for name in ("control_id", "pair_id", "statistical_identity"):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        if not isinstance(self.metric_record_cell_id, str) or not self.metric_record_cell_id:
            raise SignalControlError("metric_record_cell_id must be non-empty")
        for name in (
            "pair_digest",
            "preregistration_digest",
            "metric_protocol_digest",
            "measurement_protocol_digest",
            "membership_evidence_digest",
            "source_metric_record_digest",
            "pair_metric_record_digest",
            "pair_filter_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("pair_query_count", "pair_source_count"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.pair_query_count != 2:
            raise SignalControlError(
                "pair-control result must contain exactly two preregistered queries"
            )
        metric_ids = tuple(sorted(_safe_id(item, "metric_id") for item in self.metric_ids))
        if not metric_ids or len(set(metric_ids)) != len(metric_ids):
            raise SignalControlError("pair-control result metric IDs must be unique")
        values: dict[str, float] = {}
        for metric_id, value in sorted(self.metric_values.items()):
            if metric_id not in metric_ids:
                raise SignalControlError(
                    "pair-control result contains an unregistered metric"
                )
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise SignalControlError("pair-control metric value must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise SignalControlError("pair-control metric value must be finite")
            values[metric_id] = number
        if set(values) != set(metric_ids):
            raise SignalControlError(
                "pair-control result must contain every preregistered metric"
            )
        object.__setattr__(self, "metric_ids", metric_ids)
        object.__setattr__(self, "metric_values", MappingProxyType(values))
        expected = sha256_json(self._payload_without_digest())
        if self.result_digest is None:
            object.__setattr__(self, "result_digest", expected)
        elif _digest(self.result_digest, "result_digest") != expected:
            raise SignalControlError("pair-control result digest mismatch")

    @property
    def adds_input_view(self) -> bool:
        return False

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.control_id,
            "pair_id": self.pair_id,
            "pair_digest": self.pair_digest,
            "preregistration_digest": self.preregistration_digest,
            "statistical_identity": self.statistical_identity,
            "metric_protocol_digest": self.metric_protocol_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "membership_evidence_digest": self.membership_evidence_digest,
            "source_metric_record_digest": self.source_metric_record_digest,
            "pair_metric_record_digest": self.pair_metric_record_digest,
            "pair_filter_digest": self.pair_filter_digest,
            "metric_record_cell_id": self.metric_record_cell_id,
            "metric_ids": list(self.metric_ids),
            "metric_values": dict(self.metric_values),
            "pair_query_count": self.pair_query_count,
            "pair_source_count": self.pair_source_count,
            "control_level": "PAIR_OR_BANK_RESULT",
            "adds_input_view": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "result_digest": self.result_digest}

    @property
    def metric_record_digest(self) -> str:
        """Backward-compatible name for the pair-specific metric record.

        The former evaluator pointed this field at the full atlas matrix.  It
        now intentionally aliases the filtered/recomputed record so callers
        cannot accidentally cite a global metric as pair evidence.
        """

        return self.pair_metric_record_digest

    def to_public_dict(self) -> dict[str, Any]:
        """Aggregate-only projection; pair IDs, bank IDs and rows stay private."""

        payload = {
            "schema": PUBLIC_PAIR_CONTROL_RESULT_SCHEMA,
            "control_id": self.control_id,
            "pair_digest": self.pair_digest,
            "preregistration_digest": self.preregistration_digest,
            "statistical_identity": self.statistical_identity,
            "metric_protocol_digest": self.metric_protocol_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "membership_evidence_digest": self.membership_evidence_digest,
            "source_metric_record_digest": self.source_metric_record_digest,
            "pair_metric_record_digest": self.pair_metric_record_digest,
            "pair_filter_digest": self.pair_filter_digest,
            "metric_record_cell_id": self.metric_record_cell_id,
            "metric_ids": list(self.metric_ids),
            "metric_values": dict(self.metric_values),
            "pair_query_count": self.pair_query_count,
            "pair_source_count": self.pair_source_count,
            "private_pair_membership_withheld": True,
            "private_result_digest": self.result_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def _query_receipts_by_bank(
    record: SignalMetricRecord,
) -> Mapping[str, frozenset[str]]:
    receipts: dict[str, set[str]] = {}
    for row in record.rows:
        receipts.setdefault(row.query_bank_id, set()).add(row.query_receipt_digest)
    return MappingProxyType(
        {bank_id: frozenset(values) for bank_id, values in sorted(receipts.items())}
    )


def filter_pair_control_metric_record(
    contract: PairControlContract,
    metric_record: SignalMetricRecord,
    membership: PairControlMembershipEvidence,
) -> tuple[SignalMetricRecord, str]:
    """Select exactly the preregistered pair and recompute its metrics.

    A pair control is a two-query subset evaluated against the *same complete
    source ranking* as its parent cell.  Source rows are not narrowed because
    doing so would silently change the retrieval problem.  All unrelated
    queries are removed, the query manifest receives a pair-specific digest,
    and ``SignalMetricRecord`` recomputes every metric from those rows.
    """

    if not isinstance(
        contract, (SchemaCollisionPairContract, ExactRepeatPairContract)
    ):
        raise SignalControlError("pair filter requires a typed pair contract")
    if not isinstance(metric_record, SignalMetricRecord):
        raise SignalControlError("pair filter requires a SignalMetricRecord")
    if not isinstance(membership, PairControlMembershipEvidence):
        raise SignalControlError("pair filter requires typed membership evidence")
    pair_query_ids = (contract.left.bank_id, contract.right.bank_id)
    if len(set(pair_query_ids)) != 2:
        raise SignalControlError("pair filter requires two distinct query banks")
    rows = tuple(
        row for row in metric_record.rows if row.query_bank_id in pair_query_ids
    )
    observed_queries = {row.query_bank_id for row in rows}
    if observed_queries != set(pair_query_ids):
        missing = sorted(set(pair_query_ids) - observed_queries)
        raise SignalControlError(
            f"pair-control query banks are absent from metric record: {missing!r}"
        )
    expected = {
        query_id: metric_record.expected_source_by_query[query_id]
        for query_id in pair_query_ids
        if query_id in metric_record.expected_source_by_query
    }
    if set(expected) != set(pair_query_ids):
        raise SignalControlError(
            "pair-control query membership differs from expected-source mapping"
        )
    source_sets = {
        tuple(sorted(row.source_bank_id for row in rows if row.query_bank_id == query_id))
        for query_id in pair_query_ids
    }
    if len(source_sets) != 1 or not next(iter(source_sets), ()):
        raise SignalControlError(
            "pair-control queries do not retain one complete source ranking"
        )
    parent_query_sources = {
        query_id: tuple(
            sorted(
                row.source_bank_id
                for row in metric_record.rows
                if row.query_bank_id == query_id
            )
        )
        for query_id in pair_query_ids
    }
    if any(source_ids != next(iter(source_sets)) for source_ids in parent_query_sources.values()):
        raise SignalControlError(
            "pair-control filtering changed the parent source ranking"
        )
    pair_filter_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-pair-control-filter.v0",
            "pair_digest": contract.pair_digest,
            "membership_evidence_digest": membership.evidence_digest,
            "source_metric_record_digest": metric_record.record_digest,
            "query_bank_ids": sorted(pair_query_ids),
            "query_receipt_digests": sorted(
                {row.query_receipt_digest for row in rows}
            ),
            "expected_source_by_query": dict(sorted(expected.items())),
            "source_bank_ids": list(next(iter(source_sets))),
        }
    )
    pair_record = SignalMetricRecord(
        cell_id=metric_record.cell_id,
        view_or_condition_id=metric_record.view_or_condition_id,
        representation_id=metric_record.representation_id,
        representation_coordinate_digest=(
            metric_record.representation_coordinate_digest
        ),
        representation_seed=metric_record.representation_seed,
        source_index_digest=metric_record.source_index_digest,
        query_manifest_digest=pair_filter_digest,
        rows=rows,
        expected_source_by_query=expected,
    )
    return pair_record, pair_filter_digest


def evaluate_pair_control(
    contract: PairControlContract,
    metric_record: SignalMetricRecord,
    membership: PairControlMembershipEvidence,
) -> PairControlResult:
    """Evaluate one pair control against an exact, already-computed metric record.

    This function intentionally computes no new view or representation.  It
    only closes the preregistration/runtime join and fails if the metric row,
    canonical receipts, bank membership, or measurement identity differ.
    """

    if isinstance(contract, ExactRepeatPairContract):
        raise SignalControlError(
            "exact-repeat requires direct repeated-bank distance evaluation"
        )
    if not isinstance(contract, SchemaCollisionPairContract):
        raise SignalControlError("pair-control evaluator requires a typed contract")
    if not isinstance(metric_record, SignalMetricRecord):
        raise SignalControlError(
            "pair-control evaluator requires a SignalMetricRecord"
        )
    if not isinstance(membership, PairControlMembershipEvidence):
        raise SignalControlError(
            "pair-control evaluator requires typed membership evidence"
        )
    expected_membership = (
        str(contract.pair_digest),
        contract.left.bank_reference_digest,
        contract.right.bank_reference_digest,
        contract.left.bank_id,
        contract.right.bank_id,
        contract.left.measurement_protocol_digest,
    )
    observed_membership = (
        membership.pair_digest,
        membership.left_bank_reference_digest,
        membership.right_bank_reference_digest,
        membership.left_bank_id,
        membership.right_bank_id,
        membership.measurement_protocol_digest,
    )
    if observed_membership != expected_membership:
        raise SignalControlError(
            "pair-control membership or measurement protocol differs from contract"
        )
    receipts_by_bank = _query_receipts_by_bank(metric_record)
    for side, bank_id, receipt_digest in (
        ("left", contract.left.bank_id, membership.left_receipt_digest),
        ("right", contract.right.bank_id, membership.right_receipt_digest),
    ):
        observed_receipts = receipts_by_bank.get(bank_id)
        if observed_receipts is None:
            raise SignalControlError(
                f"pair-control {side} bank is absent from metric record membership"
            )
        if observed_receipts != frozenset({receipt_digest}):
            raise SignalControlError(
                f"pair-control {side} canonical receipt differs from metric record"
            )
    pair_record, pair_filter_digest = filter_pair_control_metric_record(
        contract, metric_record, membership
    )
    metric_values = pair_record.metric_values or {}
    missing = tuple(
        metric_id for metric_id in contract.metric_ids if metric_id not in metric_values
    )
    if missing:
        raise SignalControlError(
            f"pair-control metric record is missing preregistered metrics: {missing}"
        )
    return PairControlResult(
        control_id=contract.control_id,
        pair_id=contract.pair_id,
        pair_digest=str(contract.pair_digest),
        preregistration_digest=contract.preregistration_digest,
        statistical_identity=contract.statistical_identity,
        metric_protocol_digest=contract.metric_protocol_digest,
        measurement_protocol_digest=membership.measurement_protocol_digest,
        membership_evidence_digest=str(membership.evidence_digest),
        source_metric_record_digest=metric_record.record_digest,
        pair_metric_record_digest=pair_record.record_digest,
        pair_filter_digest=pair_filter_digest,
        metric_record_cell_id=metric_record.cell_id,
        metric_ids=contract.metric_ids,
        metric_values={
            metric_id: float(metric_values[metric_id])
            for metric_id in contract.metric_ids
        },
        pair_query_count=int(metric_values["query_count"]),
        pair_source_count=int(metric_values["source_count"]),
    )


@dataclass(frozen=True)
class PairControlEvaluation:
    """Private, fully replayable evidence for one pair-control result."""

    contract: PairControlContract
    membership: PairControlMembershipEvidence
    source_metric_record: SignalMetricRecord
    pair_metric_record: SignalMetricRecord
    result: PairControlResult
    evaluation_digest: str | None = None
    schema: str = PAIR_CONTROL_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_CONTROL_EVALUATION_SCHEMA:
            raise SignalControlError("unsupported PairControlEvaluation schema")
        if not isinstance(self.contract, SchemaCollisionPairContract):
            raise SignalControlError(
                "subset pair evaluation applies only to schema-collision controls"
            )
        if not isinstance(self.membership, PairControlMembershipEvidence):
            raise SignalControlError("pair evaluation requires typed membership")
        if not isinstance(self.source_metric_record, SignalMetricRecord) or not isinstance(
            self.pair_metric_record, SignalMetricRecord
        ):
            raise SignalControlError("pair evaluation requires typed metric records")
        if not isinstance(self.result, PairControlResult):
            raise SignalControlError("pair evaluation requires a typed result")
        expected_record, expected_filter = filter_pair_control_metric_record(
            self.contract, self.source_metric_record, self.membership
        )
        expected_result = evaluate_pair_control(
            self.contract, self.source_metric_record, self.membership
        )
        if (
            self.pair_metric_record.to_dict() != expected_record.to_dict()
            or self.result.to_dict() != expected_result.to_dict()
            or self.result.pair_filter_digest != expected_filter
        ):
            raise SignalControlError(
                "pair evaluation is not the exact filtered/recomputed result"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.evaluation_digest is None:
            object.__setattr__(self, "evaluation_digest", expected)
        elif _digest(self.evaluation_digest, "evaluation_digest") != expected:
            raise SignalControlError("pair evaluation digest mismatch")

    @classmethod
    def evaluate(
        cls,
        contract: PairControlContract,
        metric_record: SignalMetricRecord,
        membership: PairControlMembershipEvidence,
    ) -> "PairControlEvaluation":
        if not isinstance(contract, SchemaCollisionPairContract):
            raise SignalControlError(
                "exact-repeat must use ExactRepeatDistanceResult.evaluate"
            )
        pair_record, _ = filter_pair_control_metric_record(
            contract, metric_record, membership
        )
        return cls(
            contract=contract,
            membership=membership,
            source_metric_record=metric_record,
            pair_metric_record=pair_record,
            result=evaluate_pair_control(contract, metric_record, membership),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pair_digest": self.contract.pair_digest,
            "membership_evidence_digest": self.membership.evidence_digest,
            "source_metric_record_digest": self.source_metric_record.record_digest,
            "pair_metric_record_digest": self.pair_metric_record.record_digest,
            "result_digest": self.result.result_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evaluation_digest": self.evaluation_digest}

    def to_private_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "contract": self.contract.to_dict(),
            "left_bank_reference": self.contract.left.to_dict(),
            "right_bank_reference": self.contract.right.to_dict(),
            "membership": self.membership.to_dict(),
            "source_metric_record": self.source_metric_record.to_dict(),
            "pair_metric_record": self.pair_metric_record.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class ExactRepeatDistanceResult:
    """Direct repeated-bank MMD/noise-floor evidence in one frozen direction.

    ``left`` is always the query bank and ``right`` is always a source bank.
    A retrieval hit-rate is intentionally not accepted as a substitute.
    """

    contract: ExactRepeatPairContract
    membership: PairControlMembershipEvidence
    signal_cell_run: Any
    direct_row: SignalDistanceRow
    direction: str = "LEFT_QUERY_TO_RIGHT_SOURCE"
    result_digest: str | None = None
    schema: str = EXACT_REPEAT_DISTANCE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXACT_REPEAT_DISTANCE_RESULT_SCHEMA:
            raise SignalControlError(
                "unsupported ExactRepeatDistanceResult schema"
            )
        if not isinstance(self.contract, ExactRepeatPairContract):
            raise SignalControlError(
                "exact-repeat distance requires ExactRepeatPairContract"
            )
        if self.contract.metric_ids != ("direct_repeat_mmd",):
            raise SignalControlError(
                "exact-repeat metric contract must be direct_repeat_mmd"
            )
        if not isinstance(self.membership, PairControlMembershipEvidence):
            raise SignalControlError(
                "exact-repeat distance requires typed membership"
            )
        # Local import avoids a module-level cycle: signal_runtime owns
        # SignalCellRun and imports these control contracts.
        from .signal_runtime import SignalCellRun

        if not isinstance(self.signal_cell_run, SignalCellRun) or not isinstance(
            self.direct_row, SignalDistanceRow
        ):
            raise SignalControlError(
                "exact-repeat distance requires typed SignalCellRun/row"
            )
        if self.direction != "LEFT_QUERY_TO_RIGHT_SOURCE":
            raise SignalControlError("exact-repeat direction is frozen")
        expected_membership = (
            self.contract.pair_digest,
            self.contract.left.bank_reference_digest,
            self.contract.right.bank_reference_digest,
            self.contract.left.bank_id,
            self.contract.right.bank_id,
            self.contract.left.measurement_protocol_digest,
        )
        observed_membership = (
            self.membership.pair_digest,
            self.membership.left_bank_reference_digest,
            self.membership.right_bank_reference_digest,
            self.membership.left_bank_id,
            self.membership.right_bank_id,
            self.membership.measurement_protocol_digest,
        )
        if observed_membership != expected_membership:
            raise SignalControlError(
                "exact-repeat membership differs from contract"
            )
        matches = tuple(
            row
            for row in self.metric_record.rows
            if row.query_bank_id == self.contract.left.bank_id
            and row.source_bank_id == self.contract.right.bank_id
        )
        if len(matches) != 1 or matches[0].to_dict() != self.direct_row.to_dict():
            raise SignalControlError(
                "exact-repeat record lacks the frozen direct query/source row"
            )
        row = matches[0]
        expected_row_identity = (
            self.membership.left_receipt_digest,
            self.membership.right_receipt_digest,
            self.contract.left.registered_task_id,
            self.contract.right.registered_task_id,
            self.contract.left.embodiment_id,
            self.contract.right.embodiment_id,
            self.contract.left.abi_contract_id,
            self.contract.right.abi_contract_id,
            self.contract.left.goal_contract_id,
            self.contract.right.goal_contract_id,
            self.contract.left.dynamics_context_id,
            self.contract.right.dynamics_context_id,
            self.contract.left.context_id,
            self.contract.right.context_id,
        )
        observed_row_identity = (
            row.query_receipt_digest,
            row.source_receipt_digest,
            row.query_task_id,
            row.source_task_id,
            row.query_embodiment_id,
            row.source_embodiment_id,
            row.query_abi_contract_id,
            row.source_abi_contract_id,
            row.query_goal_contract_id,
            row.source_goal_contract_id,
            row.query_dynamics_context_id,
            row.source_dynamics_context_id,
            row.query_context_id,
            row.source_context_id,
        )
        if observed_row_identity != expected_row_identity:
            raise SignalControlError(
                "exact-repeat distance-row identity differs from membership"
            )
        if self.contract.left.bank_id not in self.signal_cell_run.query_run_digests:
            raise SignalControlError(
                "exact-repeat left query lacks typed query-run provenance"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.result_digest is None:
            object.__setattr__(self, "result_digest", expected)
        elif _digest(self.result_digest, "result_digest") != expected:
            raise SignalControlError("exact-repeat result digest mismatch")

    @classmethod
    def evaluate(
        cls,
        contract: ExactRepeatPairContract,
        membership: PairControlMembershipEvidence,
        signal_cell_run: Any,
    ) -> "ExactRepeatDistanceResult":
        from .signal_runtime import SignalCellRun

        if not isinstance(contract, ExactRepeatPairContract) or not isinstance(
            signal_cell_run, SignalCellRun
        ):
            raise SignalControlError(
                "exact-repeat evaluator requires typed contract/SignalCellRun"
            )
        matches = tuple(
            row
            for row in signal_cell_run.metric_record.rows
            if row.query_bank_id == contract.left.bank_id
            and row.source_bank_id == contract.right.bank_id
        )
        if len(matches) != 1:
            raise SignalControlError(
                "exact-repeat requires one direct left-query/right-source row"
            )
        return cls(
            contract=contract,
            membership=membership,
            signal_cell_run=signal_cell_run,
            direct_row=matches[0],
        )

    @property
    def metric_record(self) -> SignalMetricRecord:
        return self.signal_cell_run.metric_record

    @property
    def signal_cell_run_digest(self) -> str:
        return str(self.signal_cell_run.run_digest)

    @property
    def kernel_protocol_digest(self) -> str:
        return str(self.signal_cell_run.kernel_protocol.protocol_digest)

    @property
    def query_run_digest(self) -> str:
        return str(
            self.signal_cell_run.query_run_digests[self.contract.left.bank_id]
        )

    @property
    def distance(self) -> float:
        return self.direct_row.distance

    @property
    def metric_id(self) -> str:
        return "direct_repeat_mmd"

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control_id": self.contract.control_id,
            "pair_digest": self.contract.pair_digest,
            "preregistration_digest": self.contract.preregistration_digest,
            "metric_protocol_digest": self.contract.metric_protocol_digest,
            "measurement_protocol_digest": (
                self.membership.measurement_protocol_digest
            ),
            "membership_evidence_digest": self.membership.evidence_digest,
            "metric_record_digest": self.metric_record.record_digest,
            "signal_cell_run_digest": self.signal_cell_run_digest,
            "metric_record_cell_id": self.metric_record.cell_id,
            "representation_id": self.metric_record.representation_id,
            "representation_coordinate_digest": (
                self.metric_record.representation_coordinate_digest
            ),
            "representation_seed": self.metric_record.representation_seed,
            "source_index_digest": self.metric_record.source_index_digest,
            "query_manifest_digest": self.metric_record.query_manifest_digest,
            "kernel_protocol_digest": self.kernel_protocol_digest,
            "query_run_digest": self.query_run_digest,
            "direction": self.direction,
            "direct_row_digest": sha256_json(self.direct_row.to_dict()),
            "metric_id": self.metric_id,
            "distance": self.distance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "result_digest": self.result_digest}

    def to_private_dict(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "contract": self.contract.to_dict(),
            "left_bank_reference": self.contract.left.to_dict(),
            "right_bank_reference": self.contract.right.to_dict(),
            "membership": self.membership.to_dict(),
            "signal_cell_run": self.signal_cell_run.to_dict(),
            "kernel_protocol": self.signal_cell_run.kernel_protocol.to_dict(),
            "metric_record": self.metric_record.to_dict(),
            "direct_row": self.direct_row.to_dict(),
        }

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PUBLIC_EXACT_REPEAT_DISTANCE_SCHEMA,
            "control_id": self.contract.control_id,
            "pair_digest": self.contract.pair_digest,
            "preregistration_digest": self.contract.preregistration_digest,
            "metric_protocol_digest": self.contract.metric_protocol_digest,
            "measurement_protocol_digest": (
                self.membership.measurement_protocol_digest
            ),
            "membership_evidence_digest": self.membership.evidence_digest,
            "metric_record_digest": self.metric_record.record_digest,
            "signal_cell_run_digest": self.signal_cell_run_digest,
            "metric_record_cell_id": self.metric_record.cell_id,
            "representation_id": self.metric_record.representation_id,
            "representation_coordinate_digest": (
                self.metric_record.representation_coordinate_digest
            ),
            "representation_seed": self.metric_record.representation_seed,
            "source_index_digest": self.metric_record.source_index_digest,
            "kernel_protocol_digest": self.kernel_protocol_digest,
            "direction": self.direction,
            "metric_id": self.metric_id,
            "distance": self.distance,
            "private_pair_membership_withheld": True,
            "private_result_digest": self.result_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


@dataclass(frozen=True)
class ExactRepeatNoiseRatio:
    repeat_result_digest: str
    signal_cell_run_digest: str
    kernel_protocol_digest: str
    metric_record_digest: str
    representation_coordinate_digest: str
    representation_seed: int | None
    source_index_digest: str
    between_axis_scope: str
    between_row_set_digest: str
    between_row_count: int
    between_distance: float
    repeat_distance: float
    ratio: float | None
    ratio_kind: str
    ratio_digest: str | None = None
    schema: str = EXACT_REPEAT_NOISE_RATIO_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXACT_REPEAT_NOISE_RATIO_SCHEMA:
            raise SignalControlError("unsupported ExactRepeatNoiseRatio schema")
        for name in (
            "repeat_result_digest",
            "signal_cell_run_digest",
            "kernel_protocol_digest",
            "metric_record_digest",
            "representation_coordinate_digest",
            "source_index_digest",
            "between_row_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.between_axis_scope != "SAME_TASK_GOAL_EMBODIMENT_ABI_DIFFERENT_DYNAMICS":
            raise SignalControlError(
                "noise ratio numerator axis scope is frozen"
            )
        object.__setattr__(
            self,
            "between_row_count",
            _positive_int(self.between_row_count, "between_row_count"),
        )
        for name in ("between_distance", "repeat_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SignalControlError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise SignalControlError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.repeat_distance == 0.0:
            if self.between_distance <= 0.0 or self.ratio is not None or self.ratio_kind != "INFINITE_ZERO_NOISE_FLOOR":
                raise SignalControlError(
                    "zero exact-repeat noise floor requires typed infinite ratio"
                )
        else:
            if self.ratio_kind != "FINITE" or isinstance(self.ratio, bool) or not isinstance(
                self.ratio, (int, float)
            ):
                raise SignalControlError("finite noise ratio is malformed")
            object.__setattr__(self, "ratio", float(self.ratio))
            expected_ratio = self.between_distance / self.repeat_distance
            if not math.isclose(
                float(self.ratio), expected_ratio, rel_tol=1.0e-12, abs_tol=0.0
            ):
                raise SignalControlError("exact-repeat noise ratio is inconsistent")
        expected = sha256_json(self._payload_without_digest())
        if self.ratio_digest is None:
            object.__setattr__(self, "ratio_digest", expected)
        elif _digest(self.ratio_digest, "ratio_digest") != expected:
            raise SignalControlError("exact-repeat noise ratio digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "ratio_digest"
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "ratio_digest": self.ratio_digest}


def exact_repeat_noise_ratio(
    repeat: ExactRepeatDistanceResult,
) -> ExactRepeatNoiseRatio:
    """Compute axis-scoped between-dynamics / repeated-bank distance.

    The numerator is deterministic: for the exact-repeat left query, use every
    source in the already-frozen source index with the same task, goal,
    embodiment and ABI but a strictly different registered dynamics context.
    A mere collection/context-ID change is not dynamics evidence.  Cross-task,
    cross-goal and cross-embodiment rows can never enter the numerator.
    """

    if not isinstance(repeat, ExactRepeatDistanceResult):
        raise SignalControlError("noise ratio requires typed exact-repeat result")
    base = repeat.direct_row
    between_rows = tuple(
        row
        for row in repeat.metric_record.rows
        if row.query_bank_id == base.query_bank_id
        and row.source_task_id == base.query_task_id
        and row.source_goal_contract_id == base.query_goal_contract_id
        and row.source_embodiment_id == base.query_embodiment_id
        and row.source_abi_contract_id == base.query_abi_contract_id
        and row.source_dynamics_context_id != base.query_dynamics_context_id
    )
    if not between_rows:
        raise SignalControlError(
            "exact-repeat source index lacks same-axis different-dynamics rows"
        )
    between_distance = float(np.mean([row.distance for row in between_rows]))
    ratio = (
        None
        if repeat.distance == 0.0
        else between_distance / repeat.distance
    )
    return ExactRepeatNoiseRatio(
        repeat_result_digest=str(repeat.result_digest),
        signal_cell_run_digest=repeat.signal_cell_run_digest,
        kernel_protocol_digest=repeat.kernel_protocol_digest,
        metric_record_digest=repeat.metric_record.record_digest,
        representation_coordinate_digest=(
            repeat.metric_record.representation_coordinate_digest
        ),
        representation_seed=repeat.metric_record.representation_seed,
        source_index_digest=repeat.metric_record.source_index_digest,
        between_axis_scope="SAME_TASK_GOAL_EMBODIMENT_ABI_DIFFERENT_DYNAMICS",
        between_row_set_digest=sha256_json(
            sorted(sha256_json(row.to_dict()) for row in between_rows)
        ),
        between_row_count=len(between_rows),
        between_distance=between_distance,
        repeat_distance=repeat.distance,
        ratio=ratio,
        ratio_kind=(
            "INFINITE_ZERO_NOISE_FLOOR"
            if repeat.distance == 0.0
            else "FINITE"
        ),
    )


@dataclass(frozen=True)
class PairControlPlan:
    """Private preregistration for pair controls outside the 39-cell atlas."""

    contracts: tuple[PairControlContract, ...]
    schema: str = PAIR_CONTROL_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIR_CONTROL_PLAN_SCHEMA:
            raise SignalControlError("unsupported PairControlPlan schema")
        contracts = tuple(self.contracts)
        if not contracts or not all(
            isinstance(item, (SchemaCollisionPairContract, ExactRepeatPairContract))
            for item in contracts
        ):
            raise SignalControlError("pair-control plan requires typed contracts")
        pair_ids = tuple(item.pair_id for item in contracts)
        pair_digests = tuple(str(item.pair_digest) for item in contracts)
        if len(set(pair_ids)) != len(pair_ids) or len(set(pair_digests)) != len(
            pair_digests
        ):
            raise SignalControlError("pair-control plan membership must be unique")
        if {item.control_id for item in contracts} != {
            SCHEMA_COLLISION_CONTROL_ID,
            EXACT_REPEAT_CONTROL_ID,
        }:
            raise SignalControlError(
                "formal pair-control plan requires schema-collision and exact-repeat controls"
            )
        measurements = {
            item.left.measurement_protocol_digest for item in contracts
        }
        if len(measurements) != 1:
            raise SignalControlError(
                "pair-control plan must use one frozen measurement protocol"
            )
        object.__setattr__(
            self,
            "contracts",
            tuple(sorted(contracts, key=lambda item: str(item.pair_digest))),
        )

    @property
    def measurement_protocol_digest(self) -> str:
        return self.contracts[0].left.measurement_protocol_digest

    @property
    def plan_digest(self) -> str:
        return sha256_json(self.to_dict())

    def contract(self, pair_digest: str) -> PairControlContract:
        pair_digest = _digest(pair_digest, "pair_digest")
        for contract in self.contracts:
            if contract.pair_digest == pair_digest:
                return contract
        raise SignalControlError("pair digest is absent from the frozen plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contracts": [
                {
                    "contract": item.to_dict(),
                    "left_bank_reference": item.left.to_dict(),
                    "right_bank_reference": item.right.to_dict(),
                }
                for item in self.contracts
            ],
        }


__all__ = [
    "BANK_CONTROL_REFERENCE_SCHEMA",
    "BankControlReference",
    "EXACT_REPEAT_CONTROL_ID",
    "EXACT_REPEAT_DISTANCE_RESULT_SCHEMA",
    "EXACT_REPEAT_NOISE_RATIO_SCHEMA",
    "EXACT_REPEAT_PAIR_SCHEMA",
    "ExactRepeatPairContract",
    "ExactRepeatDistanceResult",
    "ExactRepeatNoiseRatio",
    "HISTORICAL_RANDOM_TANH_CELL_ID",
    "HISTORICAL_RANDOM_TANH_ID",
    "HISTORICAL_RANDOM_TANH_RESULT_SCHEMA",
    "HISTORICAL_RANDOM_TANH_SCHEMA",
    "HistoricalRandomTanhResult",
    "HistoricalRandomTanhSpec",
    "MATCHED_RANDOM_MLP_ID",
    "MarginalPreservationAudit",
    "PAIR_CONTROL_MEMBERSHIP_EVIDENCE_SCHEMA",
    "PAIR_CONTROL_EVALUATION_SCHEMA",
    "PAIR_CONTROL_PLAN_SCHEMA",
    "PAIR_CONTROL_RESULT_SCHEMA",
    "PUBLIC_PAIR_CONTROL_RESULT_SCHEMA",
    "PUBLIC_EXACT_REPEAT_DISTANCE_SCHEMA",
    "PairControlEvaluation",
    "PairControlMembershipEvidence",
    "PairControlPlan",
    "PairControlResult",
    "RF_SHUFFLED_NEXT_CONTROL_ID",
    "RF_SHUFFLED_NEXT_RESULT_SCHEMA",
    "RF_SHUFFLED_NEXT_SCHEMA",
    "RewardFreeShuffledNextResult",
    "RewardFreeShuffledNextSpec",
    "SCHEMA_COLLISION_CONTROL_ID",
    "SCHEMA_COLLISION_PAIR_SCHEMA",
    "SchemaCollisionPairContract",
    "SignalControlError",
    "evaluate_pair_control",
    "exact_repeat_noise_ratio",
    "filter_pair_control_metric_record",
]
