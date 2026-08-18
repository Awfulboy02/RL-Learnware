"""Deploy exactly the selected policy, with no fallback or shape adapter."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import fmean, pstdev
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..policy.bundle import validate_bundle
from ..policy.loader import load_policy
from ..policy.parity import verify_golden_parity
from ..pool.registry import DeploymentRegistry, RegistryRecord
from ..reuse.selector import SelectionResult


def _dimension(schema: Any, name: str) -> int:
    if isinstance(schema, Mapping):
        value = schema[name]
    else:
        value = getattr(schema, name)
    return int(value)


@dataclass(frozen=True)
class DeploymentResult:
    selection_id: str
    protocol_id: str
    selected_opaque_id: str
    status: str
    deployment_failure: str | None
    target_observation_dim: int
    target_action_dim: int
    policy_observation_dim: int
    policy_action_dim: int
    episode_returns: tuple[float, ...]
    mean_return: float | None
    return_std: float | None
    runtime_seconds: float
    evaluation_reset_seeds: tuple[int, ...] = ()
    evaluation_policy_seeds: tuple[int, ...] = ()

    @property
    def deployable(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.deployment-result.v0",
            "selection_id": self.selection_id,
            "protocol_id": self.protocol_id,
            "selected_opaque_id": self.selected_opaque_id,
            "status": self.status,
            "deployment_failure": self.deployment_failure,
            "target_observation_dim": self.target_observation_dim,
            "target_action_dim": self.target_action_dim,
            "policy_observation_dim": self.policy_observation_dim,
            "policy_action_dim": self.policy_action_dim,
            "episode_returns": list(self.episode_returns),
            "mean_return": self.mean_return,
            "return_std": self.return_std,
            "runtime_seconds": self.runtime_seconds,
            "evaluation_reset_seeds": list(self.evaluation_reset_seeds),
            "evaluation_policy_seeds": list(self.evaluation_policy_seeds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeploymentResult":
        expected = {
            "schema",
            "selection_id",
            "protocol_id",
            "selected_opaque_id",
            "status",
            "deployment_failure",
            "target_observation_dim",
            "target_action_dim",
            "policy_observation_dim",
            "policy_action_dim",
            "episode_returns",
            "mean_return",
            "return_std",
            "runtime_seconds",
            "evaluation_reset_seeds",
            "evaluation_policy_seeds",
        }
        if set(value) != expected or value.get("schema") != "policy-learnware.deployment-result.v0":
            raise ValueError("unsupported or malformed deployment result artifact")
        result = cls(
            selection_id=str(value["selection_id"]),
            protocol_id=str(value["protocol_id"]),
            selected_opaque_id=str(value["selected_opaque_id"]),
            status=str(value["status"]),
            deployment_failure=(
                None
                if value["deployment_failure"] is None
                else str(value["deployment_failure"])
            ),
            target_observation_dim=int(value["target_observation_dim"]),
            target_action_dim=int(value["target_action_dim"]),
            policy_observation_dim=int(value["policy_observation_dim"]),
            policy_action_dim=int(value["policy_action_dim"]),
            episode_returns=tuple(float(item) for item in value["episode_returns"]),
            mean_return=(
                None if value["mean_return"] is None else float(value["mean_return"])
            ),
            return_std=(
                None if value["return_std"] is None else float(value["return_std"])
            ),
            runtime_seconds=float(value["runtime_seconds"]),
            evaluation_reset_seeds=tuple(
                int(item) for item in value["evaluation_reset_seeds"]
            ),
            evaluation_policy_seeds=tuple(
                int(item) for item in value["evaluation_policy_seeds"]
            ),
        )
        if (
            not result.selection_id
            or not result.protocol_id
            or not result.selected_opaque_id
            or result.status not in {"success", "deployment_failure"}
            or min(
                result.target_observation_dim,
                result.target_action_dim,
                result.policy_observation_dim,
                result.policy_action_dim,
            )
            <= 0
            or not math.isfinite(result.runtime_seconds)
            or result.runtime_seconds < 0.0
            or len(result.evaluation_reset_seeds)
            != len(result.evaluation_policy_seeds)
        ):
            raise ValueError("deployment result fields are invalid")
        if result.status == "success":
            if (
                result.deployment_failure is not None
                or not result.episode_returns
                or any(not math.isfinite(item) for item in result.episode_returns)
                or result.mean_return is None
                or result.return_std is None
                or result.mean_return != fmean(result.episode_returns)
                or result.return_std != pstdev(result.episode_returns)
                or (
                    result.evaluation_reset_seeds
                    and len(result.episode_returns)
                    != len(result.evaluation_reset_seeds)
                )
            ):
                raise ValueError("successful deployment result is internally inconsistent")
        elif (
            not result.deployment_failure
            or result.episode_returns
            or result.mean_return is not None
            or result.return_std is not None
        ):
            raise ValueError("failed deployment result is internally inconsistent")
        return result


def _failure(
    selection: SelectionResult,
    record: RegistryRecord,
    observation_dim: int,
    action_dim: int,
    reason: str,
    start: float,
    reset_seeds: tuple[int, ...] = (),
    policy_seeds: tuple[int, ...] = (),
) -> DeploymentResult:
    return DeploymentResult(
        selection_id=selection.selection_id,
        protocol_id=selection.protocol_id,
        selected_opaque_id=selection.selected_opaque_id,
        status="deployment_failure",
        deployment_failure=reason,
        target_observation_dim=observation_dim,
        target_action_dim=action_dim,
        policy_observation_dim=record.native_observation_dim,
        policy_action_dim=record.native_action_dim,
        episode_returns=(),
        mean_return=None,
        return_std=None,
        runtime_seconds=time.perf_counter() - start,
        evaluation_reset_seeds=reset_seeds,
        evaluation_policy_seeds=policy_seeds,
    )


def load_registered_policy(
    record: RegistryRecord,
    *,
    fpo_root: str | Path,
    parity_atol: float = 1.0e-6,
    parity_rtol: float = 1.0e-6,
) -> Any:
    """Production loader: revalidate the on-disk bundle immediately before use."""

    metadata = validate_bundle(
        record.policy_bundle,
        expected_task=record.source_task,
    )
    if metadata.bundle_digest != record.policy_bundle_digest:
        raise ValueError("registered policy bundle digest has changed")
    if (
        metadata.observation_dim != record.native_observation_dim
        or metadata.action_dim != record.native_action_dim
    ):
        raise ValueError("registered policy dimensions have changed")
    policy = load_policy(metadata, fpo_root=fpo_root)
    parity = verify_golden_parity(
        policy, metadata, atol=parity_atol, rtol=parity_rtol
    )
    if not parity.passed:
        raise ValueError(
            "registered policy failed golden parity immediately before deployment"
        )
    return policy


def deploy_selected(
    selection: SelectionResult,
    registry: DeploymentRegistry,
    target_schema: Any,
    *,
    policy_loader: Callable[[RegistryRecord], Any] | None = None,
    fpo_root: str | Path | None = None,
    evaluator: Callable[[Any], Iterable[float]],
    evaluation_reset_seeds: Iterable[int] = (),
    evaluation_policy_seeds: Iterable[int] = (),
    expected_episode_count: int | None = None,
) -> DeploymentResult:
    """Check native schema, then load/evaluate only ``selection``'s policy.

    There is intentionally no parameter for a second candidate and no return
    value is fed back to the selector.
    """

    start = time.perf_counter()
    reset_seeds = tuple(int(value) for value in evaluation_reset_seeds)
    policy_seeds = tuple(int(value) for value in evaluation_policy_seeds)
    if len(reset_seeds) != len(policy_seeds):
        raise ValueError("evaluation reset/policy seed counts differ")
    if expected_episode_count is not None and len(reset_seeds) not in {
        0,
        int(expected_episode_count),
    }:
        raise ValueError("evaluation seed count differs from registered episode count")
    if selection.pool_id != registry.pool_id:
        raise ValueError("selection and private registry pool ids differ")
    if selection.pool_digest != registry.pool_digest:
        raise ValueError("selection and private registry pool digests differ")
    record = registry.get(selection.selected_opaque_id)
    if record.protocol_id != selection.protocol_id:
        raise ValueError("selection and private registry protocol ids differ")
    observation_dim = _dimension(target_schema, "observation_dim")
    action_dim = _dimension(target_schema, "action_dim")
    if (
        observation_dim != record.native_observation_dim
        or action_dim != record.native_action_dim
    ):
        return _failure(
            selection,
            record,
            observation_dim,
            action_dim,
            "incompatible_native_schema",
            start,
            reset_seeds,
            policy_seeds,
        )

    # This is the sole loading site: retrieval has already terminated.
    if policy_loader is None:
        if fpo_root is None:
            raise ValueError("fpo_root is required when no policy_loader is injected")
        resolved_loader = lambda item: load_registered_policy(item, fpo_root=fpo_root)
    else:
        resolved_loader = policy_loader
    try:
        policy = resolved_loader(record)
    except Exception as error:
        return _failure(
            selection,
            record,
            observation_dim,
            action_dim,
            f"policy_load_failed:{type(error).__name__}",
            start,
            reset_seeds,
            policy_seeds,
        )
    if int(policy.observation_dim) != observation_dim or int(policy.action_dim) != action_dim:
        return _failure(
            selection,
            record,
            observation_dim,
            action_dim,
            "loaded_policy_schema_mismatch",
            start,
            reset_seeds,
            policy_seeds,
        )
    try:
        returns = tuple(float(value) for value in evaluator(policy))
    except Exception as error:
        return _failure(
            selection,
            record,
            observation_dim,
            action_dim,
            f"policy_evaluation_failed:{type(error).__name__}",
            start,
            reset_seeds,
            policy_seeds,
        )
    if (
        not returns
        or any(not math.isfinite(value) for value in returns)
        or (
            expected_episode_count is not None
            and len(returns) != int(expected_episode_count)
        )
        or (reset_seeds and len(returns) != len(reset_seeds))
    ):
        return _failure(
            selection,
            record,
            observation_dim,
            action_dim,
            "invalid_episode_returns",
            start,
            reset_seeds,
            policy_seeds,
        )
    return DeploymentResult(
        selection_id=selection.selection_id,
        protocol_id=selection.protocol_id,
        selected_opaque_id=selection.selected_opaque_id,
        status="success",
        deployment_failure=None,
        target_observation_dim=observation_dim,
        target_action_dim=action_dim,
        policy_observation_dim=record.native_observation_dim,
        policy_action_dim=record.native_action_dim,
        episode_returns=returns,
        mean_return=fmean(returns),
        return_std=pstdev(returns),
        runtime_seconds=time.perf_counter() - start,
        evaluation_reset_seeds=reset_seeds,
        evaluation_policy_seeds=policy_seeds,
    )
