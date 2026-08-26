"""Declarative encoder information capabilities and fail-closed checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

from ..hashing import sha256_json
from .schemas import (
    V03SchemaError,
    checked_digest,
    checked_ids,
    checked_safe_id,
    strict_mapping,
)


ENCODER_ACCESS_CARD_SCHEMA = "policy-learnware.v03-encoder-access-card.v0"
AccessTier = Literal[
    "E0_UNSUPERVISED", "E1_CATEGORICAL_SOURCE", "E2_PRIVILEGED_PARAMETER"
]
ACCESS_TIERS = frozenset(
    {"E0_UNSUPERVISED", "E1_CATEGORICAL_SOURCE", "E2_PRIVILEGED_PARAMETER"}
)

SOURCE_RAW_TRANSITIONS = "source_raw_transitions"
SOURCE_EPISODE_PROBE_BOUNDARIES = "source_episode_probe_boundaries"
SOURCE_TASK_CATEGORICAL_LABELS = "source_task_categorical_labels"
SOURCE_ANCHOR_CATEGORICAL_LABELS = "source_anchor_categorical_labels"
SOURCE_AXIS_CATEGORICAL_LABELS = "source_axis_categorical_labels"
SOURCE_NUMERIC_FACTOR_PARAMETERS = "source_numeric_factor_physical_parameters"
PROBE_STYLE_LABELS = "probe_style_labels"
REWARD_CHANNEL = "reward_channel"
DEVELOPMENT_TARGET_TRANSITIONS = "development_target_transitions"
DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS = "development_oracle_returns_rank_labels"
CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY = "candidate_rollouts_q_policy_identity"
EXTERNAL_PRETRAINED_WEIGHTS = "external_pretrained_weights"
CONFIRMATORY_TARGET_LABELS_RETURNS = "confirmatory_target_labels_returns"
CONFIRMATORY_ORACLE = "confirmatory_oracle"
CANDIDATE_TARGET_ROLLOUTS_Q = "candidate_target_rollouts_q"
TARGET_GRADIENT_UPDATE = "target_gradient_update"

DECLARABLE_CAPABILITIES = frozenset(
    {
        SOURCE_RAW_TRANSITIONS,
        SOURCE_EPISODE_PROBE_BOUNDARIES,
        SOURCE_TASK_CATEGORICAL_LABELS,
        SOURCE_ANCHOR_CATEGORICAL_LABELS,
        SOURCE_AXIS_CATEGORICAL_LABELS,
        SOURCE_NUMERIC_FACTOR_PARAMETERS,
        PROBE_STYLE_LABELS,
        REWARD_CHANNEL,
        DEVELOPMENT_TARGET_TRANSITIONS,
        DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS,
        CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY,
        EXTERNAL_PRETRAINED_WEIGHTS,
        CONFIRMATORY_TARGET_LABELS_RETURNS,
        CONFIRMATORY_ORACLE,
        CANDIDATE_TARGET_ROLLOUTS_Q,
        TARGET_GRADIENT_UPDATE,
    }
)

_E0_SOURCE = frozenset(
    {
        SOURCE_RAW_TRANSITIONS,
        SOURCE_EPISODE_PROBE_BOUNDARIES,
        REWARD_CHANNEL,
        EXTERNAL_PRETRAINED_WEIGHTS,
    }
)
_E1_SOURCE = _E0_SOURCE | {
    SOURCE_TASK_CATEGORICAL_LABELS,
    SOURCE_ANCHOR_CATEGORICAL_LABELS,
}
_E2_SOURCE = _E1_SOURCE | {
    SOURCE_AXIS_CATEGORICAL_LABELS,
    SOURCE_NUMERIC_FACTOR_PARAMETERS,
}
_DEVELOPMENT_ONLY = frozenset(
    {
        DEVELOPMENT_TARGET_TRANSITIONS,
        DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS,
        PROBE_STYLE_LABELS,
    }
)
E_TABLE_FORBIDDEN_CAPABILITIES = frozenset(
    {
        DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS,
        PROBE_STYLE_LABELS,
        CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY,
        CONFIRMATORY_TARGET_LABELS_RETURNS,
        CONFIRMATORY_ORACLE,
        CANDIDATE_TARGET_ROLLOUTS_Q,
        TARGET_GRADIENT_UPDATE,
    }
)
_ABSOLUTELY_FORBIDDEN = frozenset(
    {
        CONFIRMATORY_TARGET_LABELS_RETURNS,
        CONFIRMATORY_ORACLE,
        CANDIDATE_TARGET_ROLLOUTS_Q,
        TARGET_GRADIENT_UPDATE,
    }
)


class EncoderAccessError(V03SchemaError):
    """An encoder attempted to consume undeclared or disallowed information."""


@dataclass(frozen=True)
class EncoderAccessCard:
    encoder_id: str
    access_tier: AccessTier
    declared_capabilities: tuple[str, ...]
    external_pretrained_weights_digest: str | None
    max_hyperparameter_trials: int
    total_train_compute_hours: float
    formal_eligible: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_id", checked_safe_id(self.encoder_id, "encoder_id"))
        if self.access_tier not in ACCESS_TIERS:
            raise EncoderAccessError(f"unknown encoder access tier: {self.access_tier!r}")
        capabilities = checked_ids(
            self.declared_capabilities, "declared_capabilities", allow_empty=False
        )
        unknown = set(capabilities) - set(DECLARABLE_CAPABILITIES)
        if unknown:
            raise EncoderAccessError(f"unknown encoder capabilities: {sorted(unknown)}")
        if SOURCE_RAW_TRANSITIONS not in capabilities:
            raise EncoderAccessError("every semantic encoder must declare source_raw_transitions")
        forbidden = set(capabilities) & set(_ABSOLUTELY_FORBIDDEN)
        if forbidden:
            raise EncoderAccessError(
                f"target confirmatory/oracle/update evidence is forbidden: {sorted(forbidden)}"
            )

        tier_sources = {
            "E0_UNSUPERVISED": _E0_SOURCE,
            "E1_CATEGORICAL_SOURCE": _E1_SOURCE,
            "E2_PRIVILEGED_PARAMETER": _E2_SOURCE,
        }[self.access_tier]
        source_like = set(capabilities) - set(_DEVELOPMENT_ONLY) - {
            CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY
        }
        over_tier = source_like - set(tier_sources)
        if over_tier:
            raise EncoderAccessError(
                f"capabilities exceed {self.access_tier}: {sorted(over_tier)}"
            )
        if self.formal_eligible:
            formal_forbidden = set(capabilities) & set(E_TABLE_FORBIDDEN_CAPABILITIES)
            if formal_forbidden:
                raise EncoderAccessError(
                    f"formal E-table encoder declares forbidden evidence: {sorted(formal_forbidden)}"
                )
        if not isinstance(self.formal_eligible, bool):
            raise EncoderAccessError("formal_eligible must be boolean")

        object.__setattr__(self, "declared_capabilities", tuple(sorted(capabilities)))
        has_external = EXTERNAL_PRETRAINED_WEIGHTS in capabilities
        if has_external != (self.external_pretrained_weights_digest is not None):
            raise EncoderAccessError(
                "external_pretrained_weights capability and digest must be declared together"
            )
        if self.external_pretrained_weights_digest is not None:
            object.__setattr__(
                self,
                "external_pretrained_weights_digest",
                checked_digest(
                    self.external_pretrained_weights_digest,
                    "external_pretrained_weights_digest",
                ),
            )
        if (
            isinstance(self.max_hyperparameter_trials, bool)
            or not isinstance(self.max_hyperparameter_trials, int)
            or self.max_hyperparameter_trials < 0
        ):
            raise EncoderAccessError("max_hyperparameter_trials must be a non-negative integer")
        if isinstance(self.total_train_compute_hours, bool) or not isinstance(
            self.total_train_compute_hours, (int, float)
        ):
            raise EncoderAccessError("total_train_compute_hours must be finite and non-negative")
        hours = float(self.total_train_compute_hours)
        if not math.isfinite(hours) or hours < 0.0:
            raise EncoderAccessError("total_train_compute_hours must be finite and non-negative")
        object.__setattr__(self, "total_train_compute_hours", hours)

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": ENCODER_ACCESS_CARD_SCHEMA,
            "encoder_id": self.encoder_id,
            "access_tier": self.access_tier,
            "declared_capabilities": list(self.declared_capabilities),
            "external_pretrained_weights_digest": self.external_pretrained_weights_digest,
            "max_hyperparameter_trials": self.max_hyperparameter_trials,
            "total_train_compute_hours": self.total_train_compute_hours,
            "formal_eligible": self.formal_eligible,
        }

    @property
    def access_card_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.material_dict(), "access_card_digest": self.access_card_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderAccessCard":
        fields = {
            "schema",
            "encoder_id",
            "access_tier",
            "declared_capabilities",
            "external_pretrained_weights_digest",
            "max_hyperparameter_trials",
            "total_train_compute_hours",
            "formal_eligible",
            "access_card_digest",
        }
        data = strict_mapping(value, fields, "encoder access card")
        if data["schema"] != ENCODER_ACCESS_CARD_SCHEMA:
            raise EncoderAccessError("unknown encoder access-card schema")
        try:
            card = cls(
                encoder_id=data["encoder_id"],
                access_tier=data["access_tier"],
                declared_capabilities=tuple(data["declared_capabilities"]),
                external_pretrained_weights_digest=data[
                    "external_pretrained_weights_digest"
                ],
                max_hyperparameter_trials=data["max_hyperparameter_trials"],
                total_train_compute_hours=data["total_train_compute_hours"],
                formal_eligible=data["formal_eligible"],
            )
        except (TypeError, KeyError) as exc:
            raise EncoderAccessError("invalid encoder access-card value") from exc
        if checked_digest(data["access_card_digest"], "access_card_digest") != card.access_card_digest:
            raise EncoderAccessError("encoder access-card digest does not match payload")
        return card

    def assert_can_read(self, *capabilities: str) -> None:
        requested = set(capabilities)
        unknown = requested - set(DECLARABLE_CAPABILITIES)
        if unknown:
            raise EncoderAccessError(f"unknown requested capabilities: {sorted(unknown)}")
        undeclared = requested - set(self.declared_capabilities)
        if undeclared:
            raise EncoderAccessError(
                f"encoder {self.encoder_id!r} attempted undeclared access: {sorted(undeclared)}"
            )

    def validate_e_table(self) -> None:
        forbidden = set(self.declared_capabilities) & set(E_TABLE_FORBIDDEN_CAPABILITIES)
        if forbidden:
            raise EncoderAccessError(
                f"E-table access card contains forbidden evidence: {sorted(forbidden)}"
            )


__all__ = [
    "ACCESS_TIERS",
    "CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY",
    "CANDIDATE_TARGET_ROLLOUTS_Q",
    "CONFIRMATORY_ORACLE",
    "CONFIRMATORY_TARGET_LABELS_RETURNS",
    "DECLARABLE_CAPABILITIES",
    "DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS",
    "DEVELOPMENT_TARGET_TRANSITIONS",
    "ENCODER_ACCESS_CARD_SCHEMA",
    "E_TABLE_FORBIDDEN_CAPABILITIES",
    "EXTERNAL_PRETRAINED_WEIGHTS",
    "EncoderAccessCard",
    "EncoderAccessError",
    "PROBE_STYLE_LABELS",
    "REWARD_CHANNEL",
    "SOURCE_ANCHOR_CATEGORICAL_LABELS",
    "SOURCE_AXIS_CATEGORICAL_LABELS",
    "SOURCE_EPISODE_PROBE_BOUNDARIES",
    "SOURCE_NUMERIC_FACTOR_PARAMETERS",
    "SOURCE_RAW_TRANSITIONS",
    "SOURCE_TASK_CATEGORICAL_LABELS",
    "TARGET_GRADIENT_UPDATE",
]
