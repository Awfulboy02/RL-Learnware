"""Frozen seeds and transform identities for all v0.3 signal conditions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import sha256_json
from .signal_controls import HistoricalRandomTanhSpec, RewardFreeShuffledNextSpec
from .signal_matrix import C_RF_SHUFFLED_NEXT, CORE_INPUT_VIEW_IDS
from .transition_views import (
    SEEDED_VIEW_IDS,
    V_RANDOM_ENCODER,
    transition_view_execution_digest,
)


CONDITION_EXECUTION_PLAN_SCHEMA = "policy-learnware.v03-condition-execution-plan.v0"


class ConditionPlanError(ValueError):
    """A view/control seed or executable transform differs from the freeze."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ConditionPlanError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ConditionPlanError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


@dataclass(frozen=True)
class ConditionExecutionPlan:
    transition_view_seeds: Mapping[str, int]
    rf_shuffled_next_seed: int
    historical_seed: int
    historical_output_dim: int
    historical_protocol_digest: str
    transform_digests: Mapping[str, str] | None = None
    plan_digest: str | None = None
    schema: str = CONDITION_EXECUTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONDITION_EXECUTION_PLAN_SCHEMA:
            raise ConditionPlanError("unsupported condition execution plan")
        required_seeded = set(CORE_INPUT_VIEW_IDS) & set(SEEDED_VIEW_IDS)
        seeds = dict(sorted(self.transition_view_seeds.items()))
        if set(seeds) != required_seeded:
            raise ConditionPlanError(
                "transition view seeds must exactly cover stochastic core views"
            )
        for view_id, seed in seeds.items():
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ConditionPlanError(f"{view_id}: seed is invalid")
        for name in (
            "rf_shuffled_next_seed",
            "historical_seed",
            "historical_output_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConditionPlanError(f"{name} is invalid")
        if self.historical_output_dim <= 0:
            raise ConditionPlanError("historical_output_dim must be positive")
        object.__setattr__(
            self,
            "historical_protocol_digest",
            _digest(self.historical_protocol_digest, "historical_protocol_digest"),
        )
        expected_transforms = {
            view_id: transition_view_execution_digest(
                view_id,
                seed=(seeds[view_id] if view_id in required_seeded else None),
            )
            for view_id in CORE_INPUT_VIEW_IDS
        }
        expected_transforms[C_RF_SHUFFLED_NEXT] = str(
            RewardFreeShuffledNextSpec(self.rf_shuffled_next_seed).transform_digest
        )
        expected_transforms[V_RANDOM_ENCODER] = self.historical_protocol_digest
        expected_transforms = dict(sorted(expected_transforms.items()))
        if self.transform_digests is None:
            object.__setattr__(
                self, "transform_digests", MappingProxyType(expected_transforms)
            )
        else:
            supplied = {
                key: _digest(value, f"transform_digests[{key}]")
                for key, value in sorted(self.transform_digests.items())
            }
            if supplied != expected_transforms:
                raise ConditionPlanError("condition transform digest schedule drifted")
            object.__setattr__(self, "transform_digests", MappingProxyType(supplied))
        object.__setattr__(self, "transition_view_seeds", MappingProxyType(seeds))
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise ConditionPlanError("condition execution plan digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        historical_spec: HistoricalRandomTanhSpec,
        shuffled_next_seed: int = 101,
        shuffled_reward_seed: int = 103,
        temporal_shuffle_seed: int = 107,
        rf_shuffled_next_seed: int = 109,
    ) -> "ConditionExecutionPlan":
        if not isinstance(historical_spec, HistoricalRandomTanhSpec):
            raise ConditionPlanError("condition plan requires historical spec")
        seeded = tuple(sorted(set(CORE_INPUT_VIEW_IDS) & set(SEEDED_VIEW_IDS)))
        named = {
            "V_SHUFFLED_NEXT": shuffled_next_seed,
            "V_SHUFFLED_REWARD": shuffled_reward_seed,
            "V_TEMPORAL_SHUFFLE": temporal_shuffle_seed,
        }
        return cls(
            transition_view_seeds={view_id: named[view_id] for view_id in seeded},
            rf_shuffled_next_seed=rf_shuffled_next_seed,
            historical_seed=historical_spec.seed,
            historical_output_dim=historical_spec.output_dim,
            historical_protocol_digest=str(
                historical_spec.representation_protocol_digest
            ),
        )

    def transform_digest(self, condition_id: str) -> str:
        try:
            return str((self.transform_digests or {})[condition_id])
        except KeyError as error:
            raise ConditionPlanError(f"condition is absent from freeze: {condition_id}") from error

    def validate_feature_bank(self, bank: Any) -> None:
        condition_id = getattr(bank, "condition_id", None)
        transform_digest = getattr(bank, "condition_transform_digest", None)
        if transform_digest != self.transform_digest(condition_id):
            raise ConditionPlanError(
                "feature bank executable transform differs from condition freeze"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "transition_view_seeds": dict(self.transition_view_seeds),
            "rf_shuffled_next_seed": self.rf_shuffled_next_seed,
            "historical_seed": self.historical_seed,
            "historical_output_dim": self.historical_output_dim,
            "historical_protocol_digest": self.historical_protocol_digest,
            "transform_digests": dict(self.transform_digests or {}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}


__all__ = [
    "CONDITION_EXECUTION_PLAN_SCHEMA",
    "ConditionExecutionPlan",
    "ConditionPlanError",
]
