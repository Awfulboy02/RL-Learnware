"""Executable CPU/GPU-neutral runtime for one v0.3 signal-atlas cell.

The runtime requires canonicalization receipts at its public boundary, applies
one already-frozen representation coordinate, calibrates the Gaussian kernel
from source rows only, builds reduced source KMEs and empirical query KMEs, and
returns a complete distance matrix plus representation-local metrics.

Model fitting is intentionally outside this module.  R5/R5L trainers can be
expensive, so callers first fit them through :mod:`representation_ladder` and
then pass the frozen transform here.  This keeps the execution path usable for
dry-runs without accidentally starting the large experiment matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..rkme.reducer import ReducerConfig
from .canonicalization import CanonicalizedBankReceipt
from .condition_plan import ConditionExecutionPlan, ConditionPlanError
from .compute import JointDistanceRequest, run_joint_distance_stage, tie_break_digest
from .contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    RankingKey,
    SemanticCacheKey,
    SemanticCacheRecord,
    SemanticTransform,
    SourceRepresentationIndex,
    build_empirical_query_spec,
    build_source_reduced_spec,
    derive_reducer_digest,
)
from .representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
    FittedRepresentation,
    FormalTrainedRepresentationReceipt,
    RepresentationBatch,
    RepresentationManifest,
    RepresentationOutput,
)
from .representation_plan import (
    RepresentationExecutionPlan,
    RepresentationPlanError,
)
from .signal_controls import (
    BankControlReference,
    ExactRepeatPairContract,
    HistoricalRandomTanhResult,
    PairControlMembershipEvidence,
    RF_SHUFFLED_NEXT_CONTROL_ID,
    RewardFreeShuffledNextResult,
    SchemaCollisionPairContract,
)
from .signal_matrix import SignalCell, SignalMatrixPlan
from .signal_metrics import SignalDistanceRow, SignalMetricRecord
from .transition_views import (
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    TransitionViewResult,
    apply_transition_view,
)


FORMAL_FEATURE_BANK_SCHEMA = "policy-learnware.v03-formal-feature-bank.v0"
SIGNAL_BANK_IDENTITY_SCHEMA = "policy-learnware.v03-signal-bank-identity.v0"
SIGNAL_IDENTITY_REGISTRY_SCHEMA = "policy-learnware.v03-signal-identity-registry.v0"
SIGNAL_EXECUTION_PROTOCOL_SCHEMA = "policy-learnware.v03-signal-execution-protocol.v0"
REPRESENTED_BANK_SCHEMA = "policy-learnware.v03-represented-bank.v0"
SOURCE_KERNEL_PROTOCOL_SCHEMA = "policy-learnware.v03-source-kernel-protocol.v0"
SIGNAL_CELL_RUN_SCHEMA = "policy-learnware.v03-signal-cell-run.v0"

DEVELOPMENT_SMOKE_MODE = "DEVELOPMENT_SMOKE"
FORMAL_MODE = "FORMAL"
_DATA_FITTED_REPRESENTATION_IDS = frozenset(
    {R2_SOURCE_PCA_WHITEN, R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR}
)


class SignalRuntimeError(ValueError):
    """A formal feature, representation, kernel, KME or distance run is invalid."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalRuntimeError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result != result.lower():
        raise SignalRuntimeError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SignalRuntimeError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _matrix(value: Any, where: str) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] <= 0 or not np.all(np.isfinite(raw)):
        raise SignalRuntimeError(f"{where} must be a non-empty finite matrix")
    result = np.ascontiguousarray(raw).copy()
    result.setflags(write=False)
    return result


def _episode_offsets(receipt: CanonicalizedBankReceipt) -> np.ndarray:
    episode_id = np.asarray(receipt.batch.episode_id, dtype=np.int64)
    boundaries = np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1
    result = np.concatenate(
        [np.asarray([0], dtype=np.int64), boundaries, [episode_id.size]]
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SignalBankIdentity:
    """Registered task/goal/dynamics identity for one measured bank.

    A canonicalization receipt intentionally knows only the private task ID and
    native ABI.  Signal panels need the finer embodiment/goal/dynamics labels,
    so they are supplied as a separately frozen, receipt-bound record rather
    than inferred from a filename or bank ID.
    """

    receipt_digest: str
    bank_id: str
    task_private_id: str
    embodiment_id: str
    abi_contract_id: str
    goal_contract_id: str
    dynamics_context_id: str
    context_id: str
    measurement_protocol_digest: str
    probe_seed_digest: str
    equivalence_class_id: str | None = None
    schema: str = SIGNAL_BANK_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_BANK_IDENTITY_SCHEMA:
            raise SignalRuntimeError("unsupported SignalBankIdentity schema")
        object.__setattr__(
            self, "receipt_digest", _digest(self.receipt_digest, "receipt_digest")
        )
        object.__setattr__(
            self,
            "measurement_protocol_digest",
            _digest(self.measurement_protocol_digest, "measurement_protocol_digest"),
        )
        object.__setattr__(
            self,
            "probe_seed_digest",
            _digest(self.probe_seed_digest, "probe_seed_digest"),
        )
        for name in (
            "bank_id",
            "task_private_id",
            "embodiment_id",
            "abi_contract_id",
            "goal_contract_id",
            "dynamics_context_id",
            "context_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.equivalence_class_id is not None:
            object.__setattr__(
                self,
                "equivalence_class_id",
                _nonempty(self.equivalence_class_id, "equivalence_class_id"),
            )

    @classmethod
    def from_receipt(
        cls,
        receipt: CanonicalizedBankReceipt,
        *,
        embodiment_id: str,
        abi_contract_id: str,
        goal_contract_id: str,
        dynamics_context_id: str,
        context_id: str,
        measurement_protocol_digest: str,
        probe_seed_digest: str,
        equivalence_class_id: str | None = None,
    ) -> "SignalBankIdentity":
        if not isinstance(receipt, CanonicalizedBankReceipt):
            raise SignalRuntimeError("signal bank identity requires a canonical receipt")
        return cls(
            receipt_digest=str(receipt.receipt_digest),
            bank_id=receipt.bank_id,
            task_private_id=receipt.task_private_id,
            embodiment_id=embodiment_id,
            abi_contract_id=abi_contract_id,
            goal_contract_id=goal_contract_id,
            dynamics_context_id=dynamics_context_id,
            context_id=context_id,
            measurement_protocol_digest=measurement_protocol_digest,
            probe_seed_digest=probe_seed_digest,
            equivalence_class_id=equivalence_class_id,
        )

    @property
    def identity_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class SignalIdentityRegistry:
    """Frozen join from canonical receipts to preregistered task taxonomy."""

    taxonomy_manifest_digest: str
    identities: tuple[SignalBankIdentity, ...]
    registry_digest: str | None = None
    schema: str = SIGNAL_IDENTITY_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_IDENTITY_REGISTRY_SCHEMA:
            raise SignalRuntimeError("unsupported SignalIdentityRegistry schema")
        object.__setattr__(
            self,
            "taxonomy_manifest_digest",
            _digest(self.taxonomy_manifest_digest, "taxonomy_manifest_digest"),
        )
        identities = tuple(self.identities)
        if not identities or not all(
            isinstance(item, SignalBankIdentity) for item in identities
        ):
            raise SignalRuntimeError("identity registry requires typed identities")
        bank_ids = [item.bank_id for item in identities]
        receipts = [item.receipt_digest for item in identities]
        if len(set(bank_ids)) != len(bank_ids) or len(set(receipts)) != len(receipts):
            raise SignalRuntimeError("identity registry bank/receipt membership must be unique")
        measurement = {item.measurement_protocol_digest for item in identities}
        if len(measurement) != 1:
            raise SignalRuntimeError("one identity registry cannot mix measurement protocols")
        identities = tuple(sorted(identities, key=lambda item: item.bank_id))
        object.__setattr__(self, "identities", identities)
        expected = sha256_json(self._payload_without_digest())
        if self.registry_digest is None:
            object.__setattr__(self, "registry_digest", expected)
        elif _digest(self.registry_digest, "registry_digest") != expected:
            raise SignalRuntimeError("identity registry digest does not match contents")

    @property
    def measurement_protocol_digest(self) -> str:
        return self.identities[0].measurement_protocol_digest

    def identity_for_receipt(
        self, receipt: CanonicalizedBankReceipt
    ) -> SignalBankIdentity:
        matches = tuple(
            item
            for item in self.identities
            if item.receipt_digest == receipt.receipt_digest
            and item.bank_id == receipt.bank_id
            and item.task_private_id == receipt.task_private_id
        )
        if len(matches) != 1:
            raise SignalRuntimeError(
                "canonical receipt is absent from the frozen identity registry"
            )
        return matches[0]

    def validate_feature_bank(self, bank: "FormalFeatureBank") -> None:
        expected = self.identity_for_receipt(bank.receipt)
        if expected.to_dict() != bank.identity.to_dict():
            raise SignalRuntimeError(
                "feature-bank identity differs from the frozen taxonomy registry"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "taxonomy_manifest_digest": self.taxonomy_manifest_digest,
            "identity_digests": [item.identity_digest for item in self.identities],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "identities": [item.to_dict() for item in self.identities],
            "registry_digest": self.registry_digest,
        }


@dataclass(frozen=True)
class SignalExecutionProtocol:
    """Frozen numeric protocol and exact stochastic evaluation schedule."""

    plan_digest: str
    identity_registry_digest: str
    measurement_protocol_digest: str
    representation_plan: RepresentationExecutionPlan
    condition_plan: ConditionExecutionPlan
    execution_mode: str
    reducer_config: ReducerConfig
    pair_budget: int = 10_000
    bandwidth_seed: int = 0
    block_size: int = 2048
    representation_seeds: tuple[int, ...] = (0, 1, 2)
    historical_seed: int = 0
    protocol_digest: str | None = None
    schema: str = SIGNAL_EXECUTION_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_EXECUTION_PROTOCOL_SCHEMA:
            raise SignalRuntimeError("unsupported SignalExecutionProtocol schema")
        for name in (
            "plan_digest",
            "identity_registry_digest",
            "measurement_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.representation_plan, RepresentationExecutionPlan):
            raise SignalRuntimeError("execution protocol requires representation plan")
        if not isinstance(self.condition_plan, ConditionExecutionPlan):
            raise SignalRuntimeError("execution protocol requires condition plan")
        if (
            self.representation_plan.signal_matrix_digest != self.plan_digest
            or self.representation_plan.historical_seed != self.historical_seed
            or self.condition_plan.historical_seed != self.historical_seed
            or self.condition_plan.historical_output_dim
            != self.representation_plan.historical_output_dim
            or self.condition_plan.historical_protocol_digest
            != self.representation_plan.historical_protocol_digest
        ):
            raise SignalRuntimeError(
                "representation/condition plans differ from signal/historical freeze"
            )
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise SignalRuntimeError("execution_mode is invalid")
        if not isinstance(self.reducer_config, ReducerConfig):
            raise SignalRuntimeError("execution protocol requires ReducerConfig")
        for name, allow_zero in (
            ("pair_budget", False),
            ("bandwidth_seed", True),
            ("block_size", False),
            ("historical_seed", True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or (
                value < 0 if allow_zero else value <= 0
            ):
                raise SignalRuntimeError(f"{name} is invalid")
        seeds = tuple(self.representation_seeds)
        if seeds != (0, 1, 2):
            raise SignalRuntimeError("representation seeds must be exactly (0, 1, 2)")
        object.__setattr__(self, "representation_seeds", seeds)
        expected = sha256_json(self._payload_without_digest())
        if self.protocol_digest is None:
            object.__setattr__(self, "protocol_digest", expected)
        elif _digest(self.protocol_digest, "protocol_digest") != expected:
            raise SignalRuntimeError("execution protocol digest does not match contents")

    def expected_evaluation_seeds(self, cell: SignalCell) -> tuple[int | None, ...]:
        if cell.representation_id in {
            R1_FIXED_RANDOM_LINEAR,
            R3_MATCHED_RANDOM_MLP,
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            return self.representation_seeds
        if cell.representation_id == R_HIST_RANDOM_TANH:
            return (self.historical_seed,)
        if cell.representation_id in {R0_PADDED_RAW, R2_SOURCE_PCA_WHITEN}:
            return (None,)
        raise SignalRuntimeError(
            f"execution schedule has no representation: {cell.representation_id}"
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "identity_registry_digest": self.identity_registry_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "representation_plan_digest": self.representation_plan.plan_digest,
            "condition_plan_digest": self.condition_plan.plan_digest,
            "execution_mode": self.execution_mode,
            "reducer_digest": derive_reducer_digest(self.reducer_config),
            "query_kme_mode": "EMPIRICAL",
            "source_kme_mode": "REDUCED",
            "pair_budget": self.pair_budget,
            "bandwidth_seed": self.bandwidth_seed,
            "block_size": self.block_size,
            "representation_seeds": list(self.representation_seeds),
            "historical_seed": self.historical_seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "protocol_digest": self.protocol_digest}


@dataclass(frozen=True)
class FormalFeatureBank:
    """Feature rows whose canonicalization provenance cannot be omitted."""

    receipt: CanonicalizedBankReceipt
    identity: SignalBankIdentity
    condition_id: str
    condition_transform_digest: str
    values: np.ndarray
    rf_shuffled_next_result: RewardFreeShuffledNextResult | None = None
    feature_bank_digest: str | None = None
    schema: str = FORMAL_FEATURE_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_FEATURE_BANK_SCHEMA:
            raise SignalRuntimeError("unsupported FormalFeatureBank schema")
        if not isinstance(self.receipt, CanonicalizedBankReceipt):
            raise SignalRuntimeError(
                "formal feature construction requires CanonicalizedBankReceipt"
            )
        if not isinstance(self.identity, SignalBankIdentity):
            raise SignalRuntimeError("formal feature bank requires SignalBankIdentity")
        if (
            self.identity.receipt_digest != self.receipt.receipt_digest
            or self.identity.bank_id != self.receipt.bank_id
            or self.identity.task_private_id != self.receipt.task_private_id
        ):
            raise SignalRuntimeError(
                "signal bank identity is not bound to the canonical receipt"
            )
        object.__setattr__(self, "condition_id", _nonempty(self.condition_id, "condition_id"))
        object.__setattr__(
            self,
            "condition_transform_digest",
            _digest(self.condition_transform_digest, "condition_transform_digest"),
        )
        values = _matrix(self.values, "feature values")
        if values.shape[0] != self.receipt.batch.transition_count:
            raise SignalRuntimeError("feature rows differ from canonical transition rows")
        object.__setattr__(self, "values", values)
        if self.condition_id == RF_SHUFFLED_NEXT_CONTROL_ID:
            if not isinstance(
                self.rf_shuffled_next_result, RewardFreeShuffledNextResult
            ):
                raise SignalRuntimeError(
                    "C_RF_SHUFFLED_NEXT requires typed transform-result provenance"
                )
            _validate_rf_shuffled_next_against_receipt(
                self.receipt,
                self.rf_shuffled_next_result,
                expected_values=values,
            )
            if (
                self.condition_transform_digest
                != self.rf_shuffled_next_result.transform_digest
            ):
                raise SignalRuntimeError(
                    "C_RF_SHUFFLED_NEXT transform differs from typed result"
                )
        elif self.rf_shuffled_next_result is not None:
            raise SignalRuntimeError(
                "RF transform-result provenance cannot attach to another condition"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.feature_bank_digest is None:
            object.__setattr__(self, "feature_bank_digest", expected)
        elif _digest(self.feature_bank_digest, "feature_bank_digest") != expected:
            raise SignalRuntimeError("feature_bank_digest does not match feature rows")

    @property
    def episode_offsets(self) -> np.ndarray:
        return _episode_offsets(self.receipt)

    @property
    def canonical_view_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-formal-canonical-view.v0",
                "condition_id": self.condition_id,
                "condition_transform_digest": self.condition_transform_digest,
                "canonicalizer_digest": self.receipt.canonicalizer_digest,
                "native_shape_registry_digest": self.receipt.native_shape_registry_digest,
            }
        )

    @property
    def condition_result_digest(self) -> str | None:
        return (
            None
            if self.rf_shuffled_next_result is None
            else str(self.rf_shuffled_next_result.dataset_digest)
        )

    @property
    def condition_audit_digest(self) -> str | None:
        return (
            None
            if self.rf_shuffled_next_result is None
            else self.rf_shuffled_next_result.marginal_audit.audit_digest
        )

    @property
    def condition_audit_passed(self) -> bool | None:
        return (
            None
            if self.rf_shuffled_next_result is None
            else self.rf_shuffled_next_result.marginal_audit.passed
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_digest": self.receipt.receipt_digest,
            "signal_bank_identity_digest": self.identity.identity_digest,
            "condition_id": self.condition_id,
            "condition_transform_digest": self.condition_transform_digest,
            "condition_result_digest": self.condition_result_digest,
            "condition_audit_digest": self.condition_audit_digest,
            "condition_audit_passed": self.condition_audit_passed,
            "canonical_view_digest": self.canonical_view_digest,
            "arrays_digest": sha256_ndarrays(
                {"values": self.values, "episode_offsets": self.episode_offsets}
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "feature_bank_digest": self.feature_bank_digest}


def _validate_rf_shuffled_next_against_receipt(
    receipt: CanonicalizedBankReceipt,
    control: RewardFreeShuffledNextResult,
    *,
    expected_values: np.ndarray | None = None,
) -> None:
    """Replay the typed control join against the canonical receipt arrays."""

    if not isinstance(receipt, CanonicalizedBankReceipt) or not isinstance(
        control, RewardFreeShuffledNextResult
    ):
        raise SignalRuntimeError(
            "RF shuffled-next provenance requires typed receipt and result"
        )
    canonical_bank = TransitionBank.from_canonical_batch(receipt.batch)
    if control.source_bank_digest != canonical_bank.canonical_bank_digest:
        raise SignalRuntimeError(
            "RF shuffled-next control was not built from the receipt batch"
        )
    base = apply_transition_view(canonical_bank, V_REWARD_FREE_TRANSITION)
    expected_channels = {
        "observation": np.asarray(base.channels["observation"], dtype=np.float32),
        "action": np.asarray(base.channels["action"], dtype=np.float32),
        "next_observation": np.asarray(
            base.channels["next_observation"], dtype=np.float32
        )[control.next_source_indices],
    }
    if (
        control.base_view_digest != base.view_digest
        or any(
            not np.array_equal(control.channels[name], expected_channels[name])
            for name in expected_channels
        )
    ):
        raise SignalRuntimeError(
            "RF shuffled-next typed result differs from receipt/base pairing"
        )
    # Direct equality to the canonical base and its indexed next-state rows is
    # stronger than trusting the result's marginal digests in isolation.
    if not control.marginal_audit.passed:
        raise SignalRuntimeError("RF shuffled-next marginal audit did not pass")
    if expected_values is not None and not np.array_equal(
        np.asarray(expected_values, dtype=np.float64),
        np.asarray(control.feature_matrix, dtype=np.float64),
    ):
        raise SignalRuntimeError(
            "C_RF_SHUFFLED_NEXT feature values differ from typed result"
        )


def feature_bank_from_transition_view(
    receipt: CanonicalizedBankReceipt,
    identity: SignalBankIdentity,
    view: TransitionViewResult,
) -> FormalFeatureBank:
    if not isinstance(receipt, CanonicalizedBankReceipt):
        raise SignalRuntimeError("feature bridge requires canonicalization receipt")
    if not isinstance(view, TransitionViewResult):
        raise SignalRuntimeError("feature bridge requires TransitionViewResult")
    if view.archived_dataset_digest != receipt.canonical_transition_digest:
        raise SignalRuntimeError("transition view was not built from the receipt batch")
    canonical_bank = TransitionBank.from_canonical_batch(receipt.batch)
    if view.source_bank_digest != canonical_bank.canonical_bank_digest:
        raise SignalRuntimeError(
            "transition view arrays were not built from the receipt batch"
        )
    return FormalFeatureBank(
        receipt=receipt,
        identity=identity,
        condition_id=view.view_id,
        # Canonical-view identity is the transform protocol, not this bank's
        # dataset-bound result digest.  SourceRepresentationIndex requires all
        # source banks in one condition to share the former.
        condition_transform_digest=str(view.execution_transform_digest),
        values=view.feature_matrix,
    )


def feature_bank_from_rf_shuffled_next(
    receipt: CanonicalizedBankReceipt,
    identity: SignalBankIdentity,
    control: RewardFreeShuffledNextResult,
) -> FormalFeatureBank:
    if not isinstance(receipt, CanonicalizedBankReceipt):
        raise SignalRuntimeError("RF feature bridge requires canonicalization receipt")
    if not isinstance(control, RewardFreeShuffledNextResult):
        raise SignalRuntimeError("RF feature bridge requires typed shuffled-next result")
    # Bind both typed objects to the *same* canonical arrays.  Constructor-local
    # validation is insufficient here: without this join, a valid control from
    # another bank could be paired with this receipt while retaining internally
    # valid digests on both sides.
    _validate_rf_shuffled_next_against_receipt(receipt, control)
    return FormalFeatureBank(
        receipt=receipt,
        identity=identity,
        condition_id=control.control_id,
        condition_transform_digest=control.transform_digest,
        values=control.feature_matrix,
        rf_shuffled_next_result=control,
    )


def bank_control_reference_from_feature_bank(
    bank: FormalFeatureBank,
) -> BankControlReference:
    """Project a measured feature bank into the frozen pair-control identity."""

    if not isinstance(bank, FormalFeatureBank):
        raise SignalRuntimeError("pair-control reference requires FormalFeatureBank")
    observation_masks = np.asarray(bank.receipt.batch.observation_mask)
    action_masks = np.asarray(bank.receipt.batch.action_mask)
    if observation_masks.ndim == 1:
        observation_masks = observation_masks[None, :]
    if action_masks.ndim == 1:
        action_masks = action_masks[None, :]
    observation_dims = np.sum(observation_masks, axis=1)
    action_dims = np.sum(action_masks, axis=1)
    if len(set(observation_dims.tolist())) != 1 or len(set(action_dims.tolist())) != 1:
        raise SignalRuntimeError("one bank cannot mix native ABI dimensions")
    return BankControlReference(
        bank_id=bank.receipt.bank_id,
        registered_task_id=bank.identity.task_private_id,
        embodiment_id=bank.identity.embodiment_id,
        abi_contract_id=bank.identity.abi_contract_id,
        goal_contract_id=bank.identity.goal_contract_id,
        dynamics_context_id=bank.identity.dynamics_context_id,
        context_id=bank.identity.context_id,
        observation_dim=int(observation_dims[0]),
        action_dim=int(action_dims[0]),
        bank_digest=bank.receipt.native_bank_digest,
        measurement_protocol_digest=bank.identity.measurement_protocol_digest,
        probe_seed_digest=bank.identity.probe_seed_digest,
    )


def validate_pair_control_feature_banks(
    contract: SchemaCollisionPairContract | ExactRepeatPairContract,
    left: FormalFeatureBank,
    right: FormalFeatureBank,
) -> PairControlMembershipEvidence:
    """Join a preregistered pair contract to the exact measured banks."""

    if not isinstance(contract, (SchemaCollisionPairContract, ExactRepeatPairContract)):
        raise SignalRuntimeError("pair control requires a typed pair contract")
    observed_left = bank_control_reference_from_feature_bank(left)
    observed_right = bank_control_reference_from_feature_bank(right)
    if (
        observed_left.to_dict() != contract.left.to_dict()
        or observed_right.to_dict() != contract.right.to_dict()
    ):
        raise SignalRuntimeError(
            "pair-control banks differ from preregistered membership"
        )
    return PairControlMembershipEvidence.create(
        contract,
        left_receipt_digest=str(left.receipt.receipt_digest),
        right_receipt_digest=str(right.receipt.receipt_digest),
        left_feature_bank_digest=str(left.feature_bank_digest),
        right_feature_bank_digest=str(right.feature_bank_digest),
    )


def represented_bank_from_historical_random_tanh(
    receipt: CanonicalizedBankReceipt,
    identity: SignalBankIdentity,
    base_view: TransitionViewResult,
    result: HistoricalRandomTanhResult,
    fitted: FittedRepresentation,
) -> "RepresentedBank":
    """Bridge the historical 14th control as an already represented bank.

    ``V_RANDOM_ENCODER`` is the historical output-space affine+tanh control,
    not an input view to which ``R_HIST_RANDOM_TANH`` should be applied again.
    The bridge therefore binds the FULL input, the historical result and the
    frozen coordinate, verifies the output once, and materializes a represented
    bank without a second encoding pass in the signal runtime.
    """

    if not isinstance(receipt, CanonicalizedBankReceipt):
        raise SignalRuntimeError("historical bridge requires canonical receipt")
    if not isinstance(base_view, TransitionViewResult) or base_view.view_id != V_FULL_LEGACY:
        raise SignalRuntimeError("historical bridge requires the FULL base view")
    if not isinstance(result, HistoricalRandomTanhResult):
        raise SignalRuntimeError("historical bridge requires typed historical result")
    if not isinstance(fitted, FittedRepresentation):
        raise SignalRuntimeError("historical bridge requires frozen representation")
    canonical_bank = TransitionBank.from_canonical_batch(receipt.batch)
    if (
        base_view.archived_dataset_digest != receipt.canonical_transition_digest
        or base_view.source_bank_digest != canonical_bank.canonical_bank_digest
        or result.source_bank_digest != canonical_bank.canonical_bank_digest
        or result.base_view_digest != base_view.view_digest
    ):
        raise SignalRuntimeError(
            "historical result/base view were not built from the receipt batch"
        )
    manifest = fitted.manifest
    if (
        manifest.representation_id != R_HIST_RANDOM_TANH
        or manifest.protocol_digest != result.representation_protocol_digest
        or manifest.checkpoint_digest != result.checkpoint_digest
    ):
        raise SignalRuntimeError(
            "historical result differs from the frozen representation coordinate"
        )
    feature = FormalFeatureBank(
        receipt=receipt,
        identity=identity,
        condition_id=V_RANDOM_ENCODER,
        condition_transform_digest=result.representation_protocol_digest,
        values=base_view.feature_matrix,
    )
    input_batch = RepresentationBatch(
        values=feature.values,
        dataset_digest=str(feature.feature_bank_digest),
        role="QUERY_TRANSFORM",
    )
    verified = fitted.transform(input_batch)
    historical_values = np.asarray(result.values, dtype=np.float64)
    if not np.array_equal(verified.values, historical_values):
        raise SignalRuntimeError(
            "historical result values differ from the frozen control checkpoint"
        )
    return RepresentedBank(
        feature_bank=feature,
        representation_manifest=manifest,
        values=historical_values,
        representation_output_digest=str(verified.output_digest),
    )


@dataclass(frozen=True)
class RepresentedBank:
    feature_bank: FormalFeatureBank
    representation_manifest: RepresentationManifest
    values: np.ndarray
    representation_output_digest: str
    formal_fit_receipt: FormalTrainedRepresentationReceipt | None = None
    represented_bank_digest: str | None = None
    schema: str = REPRESENTED_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTED_BANK_SCHEMA:
            raise SignalRuntimeError("unsupported RepresentedBank schema")
        if not isinstance(self.feature_bank, FormalFeatureBank):
            raise SignalRuntimeError("represented bank requires FormalFeatureBank")
        if not isinstance(self.representation_manifest, RepresentationManifest):
            raise SignalRuntimeError("represented bank requires RepresentationManifest")
        if self.formal_fit_receipt is not None:
            if not isinstance(
                self.formal_fit_receipt, FormalTrainedRepresentationReceipt
            ):
                raise SignalRuntimeError(
                    "formal_fit_receipt must be FormalTrainedRepresentationReceipt"
                )
            try:
                self.formal_fit_receipt.validate_manifest(
                    self.representation_manifest
                )
            except Exception as error:
                raise SignalRuntimeError(str(error)) from error
        values = _matrix(self.values, "represented values")
        if self.feature_bank.values.shape[1] != self.representation_manifest.input_dim:
            raise SignalRuntimeError(
                "feature input width differs from representation manifest"
            )
        if values.shape != (
            self.feature_bank.values.shape[0],
            self.representation_manifest.output_dim,
        ):
            raise SignalRuntimeError("represented values have incompatible shape")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "representation_output_digest",
            _digest(self.representation_output_digest, "representation_output_digest"),
        )
        input_batch = RepresentationBatch(
            values=self.feature_bank.values,
            dataset_digest=str(self.feature_bank.feature_bank_digest),
            role="QUERY_TRANSFORM",
        )
        try:
            RepresentationOutput(
                values=values,
                input_batch_digest=input_batch.batch_digest,
                coordinate_digest=str(self.representation_manifest.coordinate_digest),
                output_digest=self.representation_output_digest,
            )
        except Exception as error:
            raise SignalRuntimeError(
                "representation_output_digest is not bound to feature/coordinate/values"
            ) from error
        expected = sha256_json(self._payload_without_digest())
        if self.represented_bank_digest is None:
            object.__setattr__(self, "represented_bank_digest", expected)
        elif _digest(self.represented_bank_digest, "represented_bank_digest") != expected:
            raise SignalRuntimeError("represented_bank_digest does not match values")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_bank_digest": self.feature_bank.feature_bank_digest,
            "representation_coordinate_digest": self.representation_manifest.coordinate_digest,
            "representation_output_digest": self.representation_output_digest,
            "formal_fit_receipt_digest": (
                None
                if self.formal_fit_receipt is None
                else self.formal_fit_receipt.receipt_digest
            ),
            "arrays_digest": sha256_ndarrays(
                {"values": self.values, "episode_offsets": self.feature_bank.episode_offsets}
            ),
        }


def transform_feature_banks(
    fitted: FittedRepresentation,
    feature_banks: Sequence[FormalFeatureBank],
    *,
    formal_fit_receipt: FormalTrainedRepresentationReceipt | None = None,
) -> tuple[RepresentedBank, ...]:
    if not isinstance(fitted, FittedRepresentation):
        raise SignalRuntimeError("transform requires FittedRepresentation")
    banks = tuple(feature_banks)
    if not banks or not all(isinstance(item, FormalFeatureBank) for item in banks):
        raise SignalRuntimeError("transform requires typed formal feature banks")
    if len({item.condition_id for item in banks}) != 1:
        raise SignalRuntimeError("one cell run cannot mix signal conditions")
    if len({item.values.shape[1] for item in banks}) != 1:
        raise SignalRuntimeError("formal feature banks disagree on input dimension")
    results: list[RepresentedBank] = []
    for bank in banks:
        batch = RepresentationBatch(
            values=bank.values,
            dataset_digest=str(bank.feature_bank_digest),
            role="QUERY_TRANSFORM",
        )
        output = fitted.transform(batch)
        results.append(
            RepresentedBank(
                feature_bank=bank,
                representation_manifest=fitted.manifest,
                values=output.values,
                representation_output_digest=str(output.output_digest),
                formal_fit_receipt=formal_fit_receipt,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class SourceKernelProtocol:
    representation_coordinate_digest: str
    source_represented_bank_digests: tuple[str, ...]
    measurement_protocol_digest: str
    execution_protocol_digest: str
    bandwidth: float
    pair_budget: int
    seed: int
    protocol_digest: str | None = None
    schema: str = SOURCE_KERNEL_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_KERNEL_PROTOCOL_SCHEMA:
            raise SignalRuntimeError("unsupported SourceKernelProtocol schema")
        object.__setattr__(
            self,
            "representation_coordinate_digest",
            _digest(
                self.representation_coordinate_digest,
                "representation_coordinate_digest",
            ),
        )
        for name in ("measurement_protocol_digest", "execution_protocol_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        digests = tuple(sorted(_digest(item, "source_represented_bank_digests[]") for item in self.source_represented_bank_digests))
        if len(digests) < 2 or len(set(digests)) != len(digests):
            raise SignalRuntimeError("kernel calibration requires at least two unique source banks")
        if isinstance(self.bandwidth, bool) or not isinstance(self.bandwidth, (int, float)):
            raise SignalRuntimeError("bandwidth must be numeric")
        bandwidth = float(self.bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise SignalRuntimeError("bandwidth must be finite and positive")
        if isinstance(self.pair_budget, bool) or not isinstance(self.pair_budget, int) or self.pair_budget <= 0:
            raise SignalRuntimeError("pair_budget must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise SignalRuntimeError("seed must be a non-negative integer")
        object.__setattr__(self, "source_represented_bank_digests", digests)
        object.__setattr__(self, "bandwidth", bandwidth)
        expected = sha256_json(self._payload_without_digest())
        if self.protocol_digest is None:
            object.__setattr__(self, "protocol_digest", expected)
        elif _digest(self.protocol_digest, "protocol_digest") != expected:
            raise SignalRuntimeError("kernel protocol digest does not match source fit")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "source_represented_bank_digests": list(self.source_represented_bank_digests),
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "execution_protocol_digest": self.execution_protocol_digest,
            "bandwidth": self.bandwidth,
            "pair_budget": self.pair_budget,
            "seed": self.seed,
            "fit_scope": "SOURCE_ONLY",
            "estimator": "median-positive-euclidean-distance",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "protocol_digest": self.protocol_digest}


def fit_source_kernel_protocol(
    source_banks: Sequence[RepresentedBank],
    *,
    execution_protocol: SignalExecutionProtocol,
) -> SourceKernelProtocol:
    banks = tuple(source_banks)
    if len(banks) < 2 or not all(isinstance(item, RepresentedBank) for item in banks):
        raise SignalRuntimeError("source-only kernel fit requires at least two represented banks")
    coordinates = {item.representation_manifest.coordinate_digest for item in banks}
    if len(coordinates) != 1:
        raise SignalRuntimeError("source kernel fit cannot mix representation coordinates")
    if any(item.feature_bank.receipt.data_role != "source_reference_spec" for item in banks):
        raise SignalRuntimeError("kernel bandwidth may fit only source_reference_spec rows")
    if not isinstance(execution_protocol, SignalExecutionProtocol):
        raise SignalRuntimeError("kernel fit requires frozen execution protocol")
    measurements = {
        item.feature_bank.identity.measurement_protocol_digest for item in banks
    }
    if measurements != {execution_protocol.measurement_protocol_digest}:
        raise SignalRuntimeError(
            "source banks differ from the frozen measurement protocol"
        )
    pair_budget = execution_protocol.pair_budget
    seed = execution_protocol.bandwidth_seed
    points = np.concatenate([item.values for item in banks], axis=0)
    if points.shape[0] < 2:
        raise SignalRuntimeError("kernel fit requires at least two source points")
    total_pairs = points.shape[0] * (points.shape[0] - 1) // 2
    rng = np.random.default_rng(seed)
    if total_pairs <= pair_budget:
        left, right = np.triu_indices(points.shape[0], k=1)
    else:
        left = rng.integers(0, points.shape[0], size=pair_budget)
        right = rng.integers(0, points.shape[0] - 1, size=pair_budget)
        right = np.where(right >= left, right + 1, right)
    distances = np.linalg.norm(points[left] - points[right], axis=1)
    positive = distances[distances > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise SignalRuntimeError("source-only bandwidth collapsed: all source points repeat")
    bandwidth = float(np.median(positive))
    return SourceKernelProtocol(
        representation_coordinate_digest=str(next(iter(coordinates))),
        source_represented_bank_digests=tuple(
            str(item.represented_bank_digest) for item in banks
        ),
        measurement_protocol_digest=execution_protocol.measurement_protocol_digest,
        execution_protocol_digest=str(execution_protocol.protocol_digest),
        bandwidth=bandwidth,
        pair_budget=pair_budget,
        seed=seed,
    )


def _semantic_transform(manifest: RepresentationManifest) -> SemanticTransform:
    if manifest.representation_id == R0_PADDED_RAW:
        return SemanticTransform.raw_identity()
    if manifest.checkpoint_digest is None:  # pragma: no cover - manifest rejects it
        raise SignalRuntimeError("non-R0 representation lacks checkpoint binding")
    return SemanticTransform.frozen_encoder(
        encoder_implementation_digest=manifest.implementation_digest,
        checkpoint_digest=manifest.checkpoint_digest,
        semantic_output_protocol_digest=manifest.protocol_digest,
    )


def _cache(bank: RepresentedBank) -> SemanticCacheRecord:
    feature = bank.feature_bank
    receipt = feature.receipt
    key = SemanticCacheKey(
        raw_dataset_digest=receipt.raw_dataset_digest,
        ordered_episode_window_digest=sha256_json(
            {
                "schema": "policy-learnware.v03-one-step-episode-partition.v0",
                "canonical_transition_digest": receipt.canonical_transition_digest,
                "canonicalizer_digest": receipt.canonicalizer_digest,
                "episode_offsets_digest": sha256_ndarrays(
                    {"episode_offsets": feature.episode_offsets}
                ),
            }
        ),
        canonical_view_digest=feature.canonical_view_digest,
        window_protocol_digest=sha256_json(
            {
                "schema": "policy-learnware.v03-one-step-unordered-window.v0",
                "window_length": 1,
                "ordering_used": False,
            }
        ),
        normalizer_digest=receipt.normalizer_digest,
        semantic_transform=_semantic_transform(bank.representation_manifest),
        mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    )
    return SemanticCacheRecord(
        key=key,
        points=bank.values,
        episode_offsets=feature.episode_offsets,
    )


@dataclass(frozen=True)
class SignalCellRun:
    plan_digest: str
    cell_id: str
    cell_digest: str
    execution_protocol_digest: str
    execution_mode: str
    source_fit_provenance_digest: str | None
    work_item_digest: str | None
    evaluation_seed: int | None
    kernel_protocol: SourceKernelProtocol
    source_index_digest: str
    query_run_digests: Mapping[str, str]
    metric_record: SignalMetricRecord
    diagnostics: Any
    run_digest: str | None = None
    schema: str = SIGNAL_CELL_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CELL_RUN_SCHEMA:
            raise SignalRuntimeError("unsupported SignalCellRun schema")
        object.__setattr__(
            self, "plan_digest", _digest(self.plan_digest, "plan_digest")
        )
        object.__setattr__(self, "cell_id", _nonempty(self.cell_id, "cell_id"))
        object.__setattr__(
            self, "cell_digest", _digest(self.cell_digest, "cell_digest")
        )
        object.__setattr__(
            self,
            "execution_protocol_digest",
            _digest(self.execution_protocol_digest, "execution_protocol_digest"),
        )
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise SignalRuntimeError("cell-run execution mode is invalid")
        if self.source_fit_provenance_digest is not None:
            object.__setattr__(
                self,
                "source_fit_provenance_digest",
                _digest(
                    self.source_fit_provenance_digest,
                    "source_fit_provenance_digest",
                ),
            )
        if self.work_item_digest is not None:
            object.__setattr__(
                self,
                "work_item_digest",
                _digest(self.work_item_digest, "work_item_digest"),
            )
        if self.execution_mode == FORMAL_MODE and self.work_item_digest is None:
            raise SignalRuntimeError(
                "formal cell run requires a frozen work-item digest"
            )
        if self.evaluation_seed is not None:
            if (
                isinstance(self.evaluation_seed, bool)
                or not isinstance(self.evaluation_seed, int)
                or self.evaluation_seed < 0
            ):
                raise SignalRuntimeError("evaluation_seed is invalid")
        if not isinstance(self.kernel_protocol, SourceKernelProtocol):
            raise SignalRuntimeError("cell run requires SourceKernelProtocol")
        if self.kernel_protocol.execution_protocol_digest != self.execution_protocol_digest:
            raise SignalRuntimeError("kernel protocol differs from cell execution protocol")
        object.__setattr__(
            self,
            "source_index_digest",
            _digest(self.source_index_digest, "source_index_digest"),
        )
        runs = {
            _nonempty(query, "query bank ID"): _digest(digest, "query run digest")
            for query, digest in sorted(self.query_run_digests.items())
        }
        if not runs:
            raise SignalRuntimeError("cell run requires query distance runs")
        if not isinstance(self.metric_record, SignalMetricRecord):
            raise SignalRuntimeError("cell run requires SignalMetricRecord")
        # Local import avoids a module-level cycle: diagnostics consume
        # RepresentedBank arrays while the runtime persists the aggregate.
        from .signal_diagnostics import SignalCellDiagnostics

        if not isinstance(self.diagnostics, SignalCellDiagnostics):
            raise SignalRuntimeError(
                "cell run requires typed representation/confusion diagnostics"
            )
        if (
            self.diagnostics.metric_record_digest
            != self.metric_record.record_digest
            or self.diagnostics.representation_coordinate_digest
            != self.metric_record.representation_coordinate_digest
        ):
            raise SignalRuntimeError("cell diagnostics differ from metric record")
        try:
            self.diagnostics.validate_metric_record(self.metric_record)
        except Exception as error:
            raise SignalRuntimeError(str(error)) from error
        if self.metric_record.cell_id != self.cell_id:
            raise SignalRuntimeError("metric record belongs to another signal cell")
        if self.metric_record.source_index_digest != self.source_index_digest:
            raise SignalRuntimeError("metric record source index differs from cell run")
        if self.metric_record.representation_seed != self.evaluation_seed:
            raise SignalRuntimeError("metric record seed differs from cell run")
        if (
            self.execution_mode == FORMAL_MODE
            and self.metric_record.representation_id
            in _DATA_FITTED_REPRESENTATION_IDS
            and self.source_fit_provenance_digest is None
        ):
            raise SignalRuntimeError(
                "formal data-fitted cell run requires source-fit provenance"
            )
        object.__setattr__(self, "query_run_digests", MappingProxyType(runs))
        expected = sha256_json(self._payload_without_digest())
        if self.run_digest is None:
            object.__setattr__(self, "run_digest", expected)
        elif _digest(self.run_digest, "run_digest") != expected:
            raise SignalRuntimeError("cell run digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "cell_id": self.cell_id,
            "cell_digest": self.cell_digest,
            "execution_protocol_digest": self.execution_protocol_digest,
            "execution_mode": self.execution_mode,
            "source_fit_provenance_digest": self.source_fit_provenance_digest,
            "work_item_digest": self.work_item_digest,
            "evaluation_seed": self.evaluation_seed,
            "kernel_protocol_digest": self.kernel_protocol.protocol_digest,
            "source_index_digest": self.source_index_digest,
            "query_run_digests": dict(self.query_run_digests),
            "metric_record_digest": self.metric_record.record_digest,
            "diagnostics_digest": self.diagnostics.diagnostics_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "run_digest": self.run_digest}


def run_signal_cell(
    *,
    plan: SignalMatrixPlan,
    cell: SignalCell,
    source_banks: Sequence[RepresentedBank],
    query_banks: Sequence[RepresentedBank],
    expected_source_by_query: Mapping[str, str],
    identity_registry: SignalIdentityRegistry,
    execution_protocol: SignalExecutionProtocol,
    source_fit_provenance_digest: str | None = None,
    work_item_digest: str | None = None,
) -> SignalCellRun:
    """Run one numeric signal cell without fitting a representation."""

    if not isinstance(plan, SignalMatrixPlan) or not isinstance(cell, SignalCell):
        raise SignalRuntimeError("signal cell execution requires a typed frozen plan/cell")
    try:
        frozen_cell = plan.cell(cell.cell_id)
    except Exception as error:
        raise SignalRuntimeError("signal cell is absent from the frozen plan") from error
    if frozen_cell.to_dict() != cell.to_dict():
        raise SignalRuntimeError("signal cell differs from the frozen plan entry")
    if cell.applicability != "NUMERIC":
        raise SignalRuntimeError("structural N/A cells cannot enter numeric execution")
    if not isinstance(identity_registry, SignalIdentityRegistry) or not isinstance(
        execution_protocol, SignalExecutionProtocol
    ):
        raise SignalRuntimeError(
            "formal signal execution requires identity registry and execution protocol"
        )
    if (
        execution_protocol.plan_digest != plan.plan_digest
        or execution_protocol.identity_registry_digest
        != identity_registry.registry_digest
        or execution_protocol.measurement_protocol_digest
        != identity_registry.measurement_protocol_digest
    ):
        raise SignalRuntimeError(
            "execution protocol differs from plan/identity/measurement freeze"
        )
    sources = tuple(source_banks)
    queries = tuple(query_banks)
    if len(sources) < 2 or not queries:
        raise SignalRuntimeError("signal cell requires at least two sources and one query")
    if any(item.feature_bank.receipt.data_role != "source_reference_spec" for item in sources):
        raise SignalRuntimeError("source banks must have source_reference_spec role")
    if any(item.feature_bank.receipt.data_role not in {"development_query", "confirmatory_query"} for item in queries):
        raise SignalRuntimeError("query banks must have development/confirmatory query roles")
    all_banks = (*sources, *queries)
    for item in all_banks:
        identity_registry.validate_feature_bank(item.feature_bank)
        try:
            execution_protocol.condition_plan.validate_feature_bank(
                item.feature_bank
            )
            execution_protocol.representation_plan.validate_manifest(
                item.representation_manifest
            )
        except (ConditionPlanError, RepresentationPlanError) as error:
            raise SignalRuntimeError(str(error)) from error
    coordinates = {item.representation_manifest.coordinate_digest for item in all_banks}
    conditions = {item.feature_bank.condition_id for item in all_banks}
    canonicalizers = {item.feature_bank.receipt.canonicalizer_digest for item in all_banks}
    measurement_protocols = {
        item.feature_bank.identity.measurement_protocol_digest for item in all_banks
    }
    if len(coordinates) != 1 or len(conditions) != 1 or len(canonicalizers) != 1:
        raise SignalRuntimeError(
            "signal cell cannot mix representation, condition, or canonicalizer coordinates"
        )
    if len(measurement_protocols) != 1:
        raise SignalRuntimeError("signal cell cannot mix measurement protocols")
    condition = str(next(iter(conditions)))
    representation_id = sources[0].representation_manifest.representation_id
    if condition != cell.condition_id or representation_id != cell.representation_id:
        raise SignalRuntimeError(
            "represented banks do not match the frozen signal cell identity"
        )
    if source_fit_provenance_digest is not None:
        source_fit_provenance_digest = _digest(
            source_fit_provenance_digest, "source_fit_provenance_digest"
        )
    if work_item_digest is not None:
        work_item_digest = _digest(work_item_digest, "work_item_digest")
    if execution_protocol.execution_mode == FORMAL_MODE:
        if work_item_digest is None:
            raise SignalRuntimeError(
                "formal signal execution requires a frozen work-item digest"
            )
        if (
            representation_id in _DATA_FITTED_REPRESENTATION_IDS
            and source_fit_provenance_digest is None
        ):
            raise SignalRuntimeError(
                "formal data-fitted signal execution requires source-fit provenance"
            )
        if representation_id in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            receipts = tuple(item.formal_fit_receipt for item in all_banks)
            if any(item is None for item in receipts) or len(
                {item.receipt_digest for item in receipts if item is not None}
            ) != 1:
                raise SignalRuntimeError(
                    "formal R5/R5L execution requires one shared checkpoint receipt"
                )
            receipt = receipts[0]
            assert isinstance(receipt, FormalTrainedRepresentationReceipt)
            if (
                receipt.representation_execution_plan_digest
                != execution_protocol.representation_plan.plan_digest
                or receipt.formal_source_fit_batch_digest
                != source_fit_provenance_digest
            ):
                raise SignalRuntimeError(
                    "formal checkpoint receipt differs from execution/source-fit freeze"
                )
    manifest_seeds = {item.representation_manifest.seed for item in all_banks}
    if len(manifest_seeds) != 1:
        raise SignalRuntimeError("signal cell cannot mix representation seeds")
    evaluation_seed = next(iter(manifest_seeds))
    if evaluation_seed not in execution_protocol.expected_evaluation_seeds(cell):
        raise SignalRuntimeError(
            "representation seed is absent from the frozen evaluation schedule"
        )
    kernel_protocol = fit_source_kernel_protocol(
        sources, execution_protocol=execution_protocol
    )
    measurement_protocol_id = str(execution_protocol.protocol_digest)
    source_specs = {}
    for bank in sources:
        cache = _cache(bank)
        source_id = bank.feature_bank.receipt.bank_id
        if source_id in source_specs:
            raise SignalRuntimeError("duplicate source bank ID")
        source_specs[source_id] = build_source_reduced_spec(
            cache,
            kernel_bandwidth=kernel_protocol.bandwidth,
            measurement_protocol_id=measurement_protocol_id,
            probe_dataset_digest=bank.feature_bank.receipt.raw_dataset_digest,
            reducer_config=execution_protocol.reducer_config,
        )
    index = SourceRepresentationIndex(
        representation_protocol_id=next(iter(source_specs.values())).representation_protocol_id,
        entries=source_specs,
    )
    tokens = {
        source_id: sha256_json(
            {
                "schema": "policy-learnware.v03-signal-source-tie-token.v0",
                "cell_id": cell.cell_id,
                "source_bank_id": source_id,
            }
        )
        for source_id in source_specs
    }
    selector_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-signal-distance-selector.v0",
            "formula": "ascending_empirical_to_reduced_mmd",
            "tie_break_digest": tie_break_digest(tokens),
        }
    )
    distance_rows: list[SignalDistanceRow] = []
    run_digests: dict[str, str] = {}
    source_by_id = {item.feature_bank.receipt.bank_id: item for item in sources}
    for bank in queries:
        query_id = bank.feature_bank.receipt.bank_id
        cache = _cache(bank)
        query = build_empirical_query_spec(
            cache,
            kernel_bandwidth=kernel_protocol.bandwidth,
            measurement_protocol_id=measurement_protocol_id,
            probe_dataset_digest=bank.feature_bank.receipt.raw_dataset_digest,
            episode_count=bank.feature_bank.episode_offsets.size - 1,
        )
        ranking_key = RankingKey(
            query_spec_digest=query.query_spec_digest,
            representation_index_digest=index.representation_index_digest,
            selector_digest=selector_digest,
            tie_break_digest=tie_break_digest(tokens),
        )
        request = JointDistanceRequest(
            query_spec=query,
            source_index=index,
            ranking_key=ranking_key,
            tie_break_tokens=tokens,
            block_size=execution_protocol.block_size,
        )
        run = run_joint_distance_stage(request)
        run_digests[query_id] = str(run.run_digest)
        for row in run.rows:
            source = source_by_id[row.opaque_learnware_id]
            distance_rows.append(
                SignalDistanceRow(
                    query_bank_id=query_id,
                    source_bank_id=row.opaque_learnware_id,
                    query_receipt_digest=str(bank.feature_bank.receipt.receipt_digest),
                    source_receipt_digest=str(
                        source.feature_bank.receipt.receipt_digest
                    ),
                    query_raw_dataset_digest=bank.feature_bank.receipt.raw_dataset_digest,
                    source_raw_dataset_digest=(
                        source.feature_bank.receipt.raw_dataset_digest
                    ),
                    query_task_id=bank.feature_bank.receipt.task_private_id,
                    source_task_id=source.feature_bank.receipt.task_private_id,
                    query_context_id=bank.feature_bank.identity.context_id,
                    source_context_id=source.feature_bank.identity.context_id,
                    query_embodiment_id=bank.feature_bank.identity.embodiment_id,
                    source_embodiment_id=source.feature_bank.identity.embodiment_id,
                    query_abi_contract_id=bank.feature_bank.identity.abi_contract_id,
                    source_abi_contract_id=source.feature_bank.identity.abi_contract_id,
                    query_goal_contract_id=bank.feature_bank.identity.goal_contract_id,
                    source_goal_contract_id=source.feature_bank.identity.goal_contract_id,
                    query_dynamics_context_id=(
                        bank.feature_bank.identity.dynamics_context_id
                    ),
                    source_dynamics_context_id=(
                        source.feature_bank.identity.dynamics_context_id
                    ),
                    query_equivalence_class_id=(
                        bank.feature_bank.identity.equivalence_class_id
                    ),
                    source_equivalence_class_id=(
                        source.feature_bank.identity.equivalence_class_id
                    ),
                    distance=row.result.value,
                )
            )
    metric = SignalMetricRecord(
        cell_id=cell.cell_id,
        view_or_condition_id=condition,
        representation_id=representation_id,
        representation_coordinate_digest=str(next(iter(coordinates))),
        representation_seed=sources[0].representation_manifest.seed,
        source_index_digest=index.representation_index_digest,
        query_manifest_digest=sha256_json(
            {
                "schema": "policy-learnware.v03-signal-query-manifest.v0",
                "query_bank_digests": sorted(
                    str(item.represented_bank_digest) for item in queries
                ),
                "expected_source_by_query": dict(sorted(expected_source_by_query.items())),
            }
        ),
        rows=tuple(distance_rows),
        expected_source_by_query=expected_source_by_query,
    )
    from .signal_diagnostics import build_signal_cell_diagnostics

    diagnostics = build_signal_cell_diagnostics(
        source_banks=sources,
        query_banks=queries,
        metric_record=metric,
    )
    return SignalCellRun(
        plan_digest=str(plan.plan_digest),
        cell_id=cell.cell_id,
        cell_digest=str(cell.cell_digest),
        execution_protocol_digest=str(execution_protocol.protocol_digest),
        execution_mode=execution_protocol.execution_mode,
        source_fit_provenance_digest=source_fit_provenance_digest,
        work_item_digest=work_item_digest,
        evaluation_seed=evaluation_seed,
        kernel_protocol=kernel_protocol,
        source_index_digest=index.representation_index_digest,
        query_run_digests=run_digests,
        metric_record=metric,
        diagnostics=diagnostics,
    )


__all__ = [
    "FORMAL_FEATURE_BANK_SCHEMA",
    "REPRESENTED_BANK_SCHEMA",
    "SIGNAL_BANK_IDENTITY_SCHEMA",
    "SIGNAL_CELL_RUN_SCHEMA",
    "SIGNAL_EXECUTION_PROTOCOL_SCHEMA",
    "SIGNAL_IDENTITY_REGISTRY_SCHEMA",
    "SOURCE_KERNEL_PROTOCOL_SCHEMA",
    "FormalFeatureBank",
    "RepresentedBank",
    "SignalBankIdentity",
    "SignalCellRun",
    "SignalExecutionProtocol",
    "SignalIdentityRegistry",
    "SignalRuntimeError",
    "SourceKernelProtocol",
    "bank_control_reference_from_feature_bank",
    "feature_bank_from_rf_shuffled_next",
    "feature_bank_from_transition_view",
    "fit_source_kernel_protocol",
    "run_signal_cell",
    "represented_bank_from_historical_random_tanh",
    "transform_feature_banks",
    "validate_pair_control_feature_banks",
]
