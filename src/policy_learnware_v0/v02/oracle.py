"""Private full-pool oracle contracts for Policy Learnware v0.2.

The public selector never sees the objects in this module.  In particular,
execution compatibility is derived from the deployment-private minimum calling
ABI only after an immutable selection has been published.  Task identity,
reward semantics, axis/factor metadata, and target-policy returns are therefore
not public filtering inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .schemas import ExecutionABIRecord


DeploymentStatus = Literal[
    "SELECTED_EXECUTABLE",
    "SELECTED_INCOMPATIBLE_ABI",
    "NO_SELECTION",
]


class OracleContractError(ValueError):
    """Private oracle evidence violates the frozen evaluation contract."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OracleContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _identifier(value, where).lower()
    if len(result) != 64:
        raise OracleContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise OracleContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise OracleContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OracleContractError(f"{where} must be finite")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OracleContractError(f"{where} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise OracleContractError(f"{where} must be a non-negative integer")
    return result


def _positive_int(value: Any, where: str) -> int:
    result = _nonnegative_int(value, where)
    if result == 0:
        raise OracleContractError(f"{where} must be positive")
    return result


def _market_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise OracleContractError("market_ids must be a sequence of opaque IDs")
    result = tuple(_identifier(value, "market_ids[]") for value in values)
    if not result or len(result) != len(set(result)):
        raise OracleContractError("market_ids must be non-empty and unique")
    return tuple(sorted(result))


def _extract_execution_abi(value: Any, where: str) -> ExecutionABIRecord:
    """Accept an ABI or a deployment-private entry carrying one.

    Supporting the latter lets callers pass ``V02PolicyMarket.deployment_private``
    directly without projecting any deployment fields into the public market.
    """

    if isinstance(value, ExecutionABIRecord):
        return value
    execution_abi = getattr(value, "execution_abi", None)
    if isinstance(execution_abi, ExecutionABIRecord):
        return execution_abi
    raise OracleContractError(f"{where} does not contain an ExecutionABIRecord")


def _abi_projection(value: ExecutionABIRecord) -> tuple[str, ...]:
    """Return exactly the minimum task-anonymous executable calling ABI."""

    return (
        value.protocol_family_id,
        value.observation_tensor_abi_digest,
        value.action_tensor_abi_digest,
        value.action_transform_id,
        value.policy_runtime_id,
        value.state_abi_id,
    )


def minimum_executable_set(
    market_ids: Sequence[str],
    deployment_registry: Mapping[str, Any],
    target_execution_abi: ExecutionABIRecord,
) -> tuple[str, ...]:
    """Compute the non-empty full-market executable set from private ABI only.

    No task, reward, semantic schema, source anchor, axis, or factor identifier
    participates.  The deployment registry must cover the full anonymous market
    exactly; silently dropping a candidate would bias the oracle skyline.
    """

    ids = _market_ids(market_ids)
    if not isinstance(deployment_registry, Mapping):
        raise OracleContractError("deployment_registry must be a mapping")
    if set(deployment_registry) != set(ids):
        raise OracleContractError(
            "deployment-private ABI registry must exactly cover the anonymous market"
        )
    if not isinstance(target_execution_abi, ExecutionABIRecord):
        raise OracleContractError("target_execution_abi has the wrong type")
    target = _abi_projection(target_execution_abi)
    executable = tuple(
        opaque_id
        for opaque_id in ids
        if _abi_projection(
            _extract_execution_abi(
                deployment_registry[opaque_id],
                f"deployment_registry[{opaque_id!r}]",
            )
        )
        == target
    )
    if not executable:
        raise OracleContractError("minimum Execution ABI executable set is empty")
    return executable


@dataclass(frozen=True)
class OracleEpisodeRow:
    """One oracle-private target-policy episode from the full executable pool."""

    opaque_query_id: str
    opaque_learnware_id: str
    episode_index: int
    reset_seed: int
    policy_seed: int
    steps: int
    raw_return: float
    normalized_return: float
    terminated: bool
    truncated: bool
    runtime_seconds: float
    private_target_instance_digest: str
    bundle_digest: str
    seed_contract_digest: str
    evaluation_protocol_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opaque_query_id",
            _identifier(self.opaque_query_id, "opaque_query_id"),
        )
        object.__setattr__(
            self,
            "opaque_learnware_id",
            _identifier(self.opaque_learnware_id, "opaque_learnware_id"),
        )
        for name in ("episode_index", "reset_seed", "policy_seed"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
        for name in ("raw_return", "normalized_return", "runtime_seconds"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.runtime_seconds < 0.0:
            raise OracleContractError("runtime_seconds cannot be negative")
        for name in ("terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise OracleContractError(f"{name} must be boolean")
        for name in (
            "private_target_instance_digest",
            "bundle_digest",
            "seed_contract_digest",
            "evaluation_protocol_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-oracle-episode-row.v0",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
        }


@dataclass(frozen=True)
class PublishedSelection:
    """Minimal immutable projection handed from public selection to the oracle."""

    method_id: str
    selection_record_digest: str
    selected_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _identifier(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "selection_record_digest",
            _digest(self.selection_record_digest, "selection_record_digest"),
        )
        if self.selected_id is not None:
            object.__setattr__(
                self, "selected_id", _identifier(self.selected_id, "selected_id")
            )

    @classmethod
    def from_selection_record(cls, record: Any) -> "PublishedSelection":
        """Project a selector record without exposing rankings to the oracle API."""

        try:
            method_id = record.method_id
            selected_id = record.selected_id
            digest = record.digest
        except AttributeError as error:
            raise OracleContractError(
                "selection record lacks method_id, selected_id, or digest"
            ) from error
        return cls(
            method_id=method_id,
            selection_record_digest=digest,
            selected_id=selected_id,
        )


@dataclass(frozen=True)
class OracleSelectionOutcome:
    method_id: str
    selection_record_digest: str
    selected_id: str | None
    deployment_status: DeploymentStatus
    selected_value: float
    regret: float
    within_executable_regret: float
    deployment_failure_regret: float
    oracle_top1_agreement: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _identifier(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "selection_record_digest",
            _digest(self.selection_record_digest, "selection_record_digest"),
        )
        if self.selected_id is not None:
            object.__setattr__(
                self, "selected_id", _identifier(self.selected_id, "selected_id")
            )
        if self.deployment_status not in {
            "SELECTED_EXECUTABLE",
            "SELECTED_INCOMPATIBLE_ABI",
            "NO_SELECTION",
        }:
            raise OracleContractError(
                f"unknown deployment status: {self.deployment_status!r}"
            )
        for name in (
            "selected_value",
            "regret",
            "within_executable_regret",
            "deployment_failure_regret",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(
            self.regret,
            self.within_executable_regret,
            self.deployment_failure_regret,
        ) < 0.0:
            raise OracleContractError("oracle regret components cannot be negative")
        if not math.isclose(
            self.regret,
            self.within_executable_regret + self.deployment_failure_regret,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise OracleContractError("oracle regret decomposition does not add up")
        if type(self.oracle_top1_agreement) is not bool:
            raise OracleContractError("oracle_top1_agreement must be boolean")

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "selection_record_digest": self.selection_record_digest,
            "selected_id": self.selected_id,
            "deployment_status": self.deployment_status,
            "selected_value": self.selected_value,
            "regret": self.regret,
            "regret_decomposition": {
                "within_executable_selection": self.within_executable_regret,
                "deployment_failure": self.deployment_failure_regret,
            },
            "oracle_top1_agreement": self.oracle_top1_agreement,
        }


@dataclass(frozen=True)
class FullPoolOracleResult:
    """Private, digest-checked values and selection outcomes for one target."""

    opaque_query_id: str
    private_target_instance_digest: str
    evaluation_protocol_id: str
    market_ids: tuple[str, ...]
    executable_ids: tuple[str, ...]
    incompatible_ids: tuple[str, ...]
    normalized_value_vector: Mapping[str, float | None]
    raw_value_vector: Mapping[str, float | None]
    episode_counts: Mapping[str, int]
    best_in_pool_ids: tuple[str, ...]
    best_in_pool_value: float
    failure_floor: float
    tie_atol: float
    candidate_paired_seeds: bool
    episode_rows_digest: str
    execution_abi_census_digest: str
    outcomes: Mapping[str, OracleSelectionOutcome]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opaque_query_id",
            _identifier(self.opaque_query_id, "opaque_query_id"),
        )
        for name in (
            "private_target_instance_digest",
            "evaluation_protocol_id",
            "episode_rows_digest",
            "execution_abi_census_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        ids = _market_ids(self.market_ids)
        executable = tuple(sorted(self.executable_ids))
        incompatible = tuple(sorted(self.incompatible_ids))
        if (
            not executable
            or set(executable) & set(incompatible)
            or set(executable) | set(incompatible) != set(ids)
        ):
            raise OracleContractError(
                "executable/incompatible IDs must partition the full market"
            )
        values = dict(self.normalized_value_vector)
        raw_values = dict(self.raw_value_vector)
        counts = dict(self.episode_counts)
        if set(values) != set(ids) or set(raw_values) != set(ids) or set(counts) != set(ids):
            raise OracleContractError("oracle vectors must exactly cover the full market")
        for opaque_id in ids:
            expected_value = opaque_id in executable
            if (values[opaque_id] is not None) != expected_value:
                raise OracleContractError("normalized value presence disagrees with ABI census")
            if (raw_values[opaque_id] is not None) != expected_value:
                raise OracleContractError("raw value presence disagrees with ABI census")
            if expected_value:
                values[opaque_id] = _finite(values[opaque_id], f"value[{opaque_id!r}]")
                raw_values[opaque_id] = _finite(
                    raw_values[opaque_id], f"raw_value[{opaque_id!r}]"
                )
                counts[opaque_id] = _positive_int(
                    counts[opaque_id], f"episode_counts[{opaque_id!r}]"
                )
            elif counts[opaque_id] != 0:
                raise OracleContractError("incompatible policies cannot have oracle episodes")
        best_value = _finite(self.best_in_pool_value, "best_in_pool_value")
        floor = _finite(self.failure_floor, "failure_floor")
        tie_atol = _finite(self.tie_atol, "tie_atol")
        if tie_atol < 0.0:
            raise OracleContractError("tie_atol cannot be negative")
        if floor > best_value + tie_atol:
            raise OracleContractError("failure_floor cannot exceed the best executable value")
        best_ids = tuple(sorted(self.best_in_pool_ids))
        if not best_ids or not set(best_ids) <= set(executable):
            raise OracleContractError("best_in_pool_ids must be a non-empty executable subset")
        expected_best = max(float(values[item]) for item in executable)
        if not math.isclose(best_value, expected_best, rel_tol=0.0, abs_tol=1.0e-12):
            raise OracleContractError("best_in_pool_value disagrees with the value vector")
        expected_ties = tuple(
            opaque_id
            for opaque_id in executable
            if math.isclose(
                float(values[opaque_id]), best_value, rel_tol=0.0, abs_tol=tie_atol
            )
        )
        if best_ids != expected_ties:
            raise OracleContractError("best_in_pool_ids disagree with the frozen tie rule")
        outcomes = dict(self.outcomes)
        for method_id, outcome in outcomes.items():
            if method_id != outcome.method_id:
                raise OracleContractError("outcome key differs from method_id")
        if type(self.candidate_paired_seeds) is not bool:
            raise OracleContractError("candidate_paired_seeds must be boolean")
        object.__setattr__(self, "market_ids", ids)
        object.__setattr__(self, "executable_ids", executable)
        object.__setattr__(self, "incompatible_ids", incompatible)
        object.__setattr__(self, "normalized_value_vector", MappingProxyType(values))
        object.__setattr__(self, "raw_value_vector", MappingProxyType(raw_values))
        object.__setattr__(self, "episode_counts", MappingProxyType(counts))
        object.__setattr__(self, "best_in_pool_ids", best_ids)
        object.__setattr__(self, "best_in_pool_value", best_value)
        object.__setattr__(self, "failure_floor", floor)
        object.__setattr__(self, "tie_atol", tie_atol)
        object.__setattr__(self, "outcomes", MappingProxyType(outcomes))

    @property
    def pool_viability(self) -> float:
        return self.best_in_pool_value

    def deployed_value(self, opaque_learnware_id: str | None) -> float:
        if opaque_learnware_id is None:
            return self.failure_floor
        opaque_id = _identifier(opaque_learnware_id, "opaque_learnware_id")
        if opaque_id not in self.normalized_value_vector:
            raise OracleContractError("selected policy is absent from the frozen market")
        value = self.normalized_value_vector[opaque_id]
        return self.failure_floor if value is None else float(value)

    def regret_for(self, opaque_learnware_id: str | None) -> float:
        regret = self.best_in_pool_value - self.deployed_value(opaque_learnware_id)
        if regret < -self.tie_atol:
            raise OracleContractError("selected value exceeds the oracle skyline")
        return max(0.0, float(regret))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-full-pool-oracle-result.v0",
            "visibility": "private-oracle-only",
            "opaque_query_id": self.opaque_query_id,
            "private_target_instance_digest": self.private_target_instance_digest,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "market_ids": list(self.market_ids),
            "executable_ids": list(self.executable_ids),
            "incompatible_ids": list(self.incompatible_ids),
            "normalized_value_vector": dict(self.normalized_value_vector),
            "raw_value_vector": dict(self.raw_value_vector),
            "episode_counts": dict(self.episode_counts),
            "best_in_pool_ids": list(self.best_in_pool_ids),
            "best_in_pool_value": self.best_in_pool_value,
            "failure_floor": self.failure_floor,
            "tie_rule": {"kind": "absolute", "atol": self.tie_atol},
            "candidate_paired_seeds": self.candidate_paired_seeds,
            "episode_rows_digest": self.episode_rows_digest,
            "execution_abi_census_digest": self.execution_abi_census_digest,
            "outcomes": {
                method_id: outcome.to_private_dict()
                for method_id, outcome in sorted(self.outcomes.items())
            },
        }


def aggregate_full_pool_oracle(
    *,
    opaque_query_id: str,
    private_target_instance_digest: str,
    evaluation_protocol_id: str,
    market_ids: Sequence[str],
    deployment_registry: Mapping[str, Any],
    target_execution_abi: ExecutionABIRecord,
    episode_rows: Sequence[OracleEpisodeRow],
    published_selections: Sequence[PublishedSelection],
    failure_floor: float,
    tie_atol: float = 0.0,
    candidate_paired_seeds: bool = True,
) -> FullPoolOracleResult:
    """Recompute a one-context private skyline directly from episode rows.

    Every ABI-executable anonymous market entry must have raw episodes; every
    incompatible entry must have none.  A selected incompatible entry receives
    the frozen failure floor with no fallback to another ranked policy.
    """

    query_id = _identifier(opaque_query_id, "opaque_query_id")
    instance_digest = _digest(
        private_target_instance_digest, "private_target_instance_digest"
    )
    protocol_id = _digest(evaluation_protocol_id, "evaluation_protocol_id")
    ids = _market_ids(market_ids)
    executable = minimum_executable_set(ids, deployment_registry, target_execution_abi)
    incompatible = tuple(sorted(set(ids) - set(executable)))
    if type(candidate_paired_seeds) is not bool:
        raise OracleContractError("candidate_paired_seeds must be boolean")
    parsed_rows: list[OracleEpisodeRow] = []
    grouped: dict[str, list[OracleEpisodeRow]] = {opaque_id: [] for opaque_id in ids}
    for row in episode_rows:
        if not isinstance(row, OracleEpisodeRow):
            raise OracleContractError("episode_rows must contain OracleEpisodeRow objects")
        if row.opaque_query_id != query_id:
            raise OracleContractError("oracle episode belongs to another target query")
        if row.private_target_instance_digest != instance_digest:
            raise OracleContractError("oracle episode target instance digest differs")
        if row.evaluation_protocol_id != protocol_id:
            raise OracleContractError("oracle episode evaluation protocol differs")
        if row.opaque_learnware_id not in grouped:
            raise OracleContractError("oracle episode policy is absent from the market")
        if row.opaque_learnware_id in incompatible:
            raise OracleContractError(
                "ABI-incompatible policy must not be executed by the oracle"
            )
        deployment_bundle = getattr(
            deployment_registry[row.opaque_learnware_id], "bundle_digest", None
        )
        if deployment_bundle is not None and row.bundle_digest != _digest(
            deployment_bundle,
            f"deployment_registry[{row.opaque_learnware_id!r}].bundle_digest",
        ):
            raise OracleContractError(
                "oracle episode bundle differs from the deployment-private registry"
            )
        grouped[row.opaque_learnware_id].append(row)
        parsed_rows.append(row)

    signatures: dict[str, tuple[tuple[int, int, int, str], ...]] = {}
    normalized: dict[str, float | None] = {}
    raw: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for opaque_id in ids:
        rows = sorted(grouped[opaque_id], key=lambda item: item.episode_index)
        indices = tuple(row.episode_index for row in rows)
        if len(indices) != len(set(indices)):
            raise OracleContractError(
                f"duplicate oracle episode index for {opaque_id!r}"
            )
        if opaque_id in executable:
            if not rows:
                raise OracleContractError(
                    f"executable policy {opaque_id!r} has no oracle episodes"
                )
            if len({row.bundle_digest for row in rows}) != 1:
                raise OracleContractError(
                    "oracle episodes for one policy contain mixed bundle digests"
                )
            signatures[opaque_id] = tuple(
                (
                    row.episode_index,
                    row.reset_seed,
                    row.policy_seed,
                    row.seed_contract_digest,
                )
                for row in rows
            )
            normalized[opaque_id] = float(
                np.mean(
                    np.asarray(
                        [row.normalized_return for row in rows], dtype=np.float64
                    )
                )
            )
            raw[opaque_id] = float(
                np.mean(np.asarray([row.raw_return for row in rows], dtype=np.float64))
            )
            counts[opaque_id] = len(rows)
        else:
            if rows:
                raise OracleContractError("incompatible policy has oracle episodes")
            normalized[opaque_id] = None
            raw[opaque_id] = None
            counts[opaque_id] = 0

    if candidate_paired_seeds:
        reference = signatures[executable[0]]
        if any(signatures[opaque_id] != reference for opaque_id in executable[1:]):
            raise OracleContractError(
                "candidate-paired oracle seeds are not exactly aligned"
            )

    best_value = max(float(normalized[opaque_id]) for opaque_id in executable)
    atol = _finite(tie_atol, "tie_atol")
    if atol < 0.0:
        raise OracleContractError("tie_atol cannot be negative")
    best_ids = tuple(
        opaque_id
        for opaque_id in executable
        if math.isclose(
            float(normalized[opaque_id]), best_value, rel_tol=0.0, abs_tol=atol
        )
    )
    floor = _finite(failure_floor, "failure_floor")
    if floor > best_value + atol:
        raise OracleContractError("failure_floor cannot exceed the oracle skyline")

    selections: dict[str, PublishedSelection] = {}
    for selection in published_selections:
        if not isinstance(selection, PublishedSelection):
            raise OracleContractError(
                "published_selections must contain PublishedSelection objects"
            )
        if selection.method_id in selections:
            raise OracleContractError("duplicate method in published selections")
        if selection.selected_id is not None and selection.selected_id not in ids:
            raise OracleContractError("published selection is absent from the market")
        selections[selection.method_id] = selection

    outcomes: dict[str, OracleSelectionOutcome] = {}
    for method_id, selection in sorted(selections.items()):
        selected_id = selection.selected_id
        if selected_id is None:
            status: DeploymentStatus = "NO_SELECTION"
            selected_value = floor
            within = 0.0
            failure = max(0.0, best_value - floor)
            top1 = False
        elif selected_id in executable:
            status = "SELECTED_EXECUTABLE"
            selected_value = float(normalized[selected_id])
            within = max(0.0, best_value - selected_value)
            failure = 0.0
            top1 = selected_id in best_ids
        else:
            status = "SELECTED_INCOMPATIBLE_ABI"
            selected_value = floor
            within = 0.0
            failure = max(0.0, best_value - floor)
            top1 = False
        outcomes[method_id] = OracleSelectionOutcome(
            method_id=method_id,
            selection_record_digest=selection.selection_record_digest,
            selected_id=selected_id,
            deployment_status=status,
            selected_value=selected_value,
            regret=within + failure,
            within_executable_regret=within,
            deployment_failure_regret=failure,
            oracle_top1_agreement=top1,
        )

    sorted_rows = sorted(
        parsed_rows,
        key=lambda item: (item.opaque_learnware_id, item.episode_index),
    )
    abi_map = {
        opaque_id: _extract_execution_abi(
            deployment_registry[opaque_id],
            f"deployment_registry[{opaque_id!r}]",
        ).digest
        for opaque_id in ids
    }
    return FullPoolOracleResult(
        opaque_query_id=query_id,
        private_target_instance_digest=instance_digest,
        evaluation_protocol_id=protocol_id,
        market_ids=ids,
        executable_ids=executable,
        incompatible_ids=incompatible,
        normalized_value_vector=normalized,
        raw_value_vector=raw,
        episode_counts=counts,
        best_in_pool_ids=best_ids,
        best_in_pool_value=best_value,
        failure_floor=floor,
        tie_atol=atol,
        candidate_paired_seeds=candidate_paired_seeds,
        episode_rows_digest=sha256_json(
            [row.to_private_dict() for row in sorted_rows]
        ),
        execution_abi_census_digest=sha256_json(
            {
                "schema": "policy-learnware.v02-execution-abi-census.v0",
                "target_execution_abi_digest": target_execution_abi.digest,
                "candidate_execution_abi_digests": abi_map,
                "executable_ids": list(executable),
            }
        ),
        outcomes=outcomes,
    )


__all__ = [
    "DeploymentStatus",
    "FullPoolOracleResult",
    "OracleContractError",
    "OracleEpisodeRow",
    "OracleSelectionOutcome",
    "PublishedSelection",
    "aggregate_full_pool_oracle",
    "minimum_executable_set",
]
