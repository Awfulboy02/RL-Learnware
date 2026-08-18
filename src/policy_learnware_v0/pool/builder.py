"""Join independent TaskSpec and policy-champion branches without leakage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..policy.bundle import PolicyBundleMetadata, validate_bundle
from ..policy.championize import CandidateEvaluation, RankedCandidate, TaskChampion
from ..hashing import sha256_json
from ..schemas import FrozenProtocol
from .learnware import LearnwarePool, PoolValidationError, SelectorEntry, SelectorTaskSpec
from .registry import DeploymentRegistry, RegistryRecord


@dataclass(frozen=True)
class BuiltPool:
    public_pool: LearnwarePool
    private_registry: DeploymentRegistry


def _metadata(champion: Any) -> PolicyBundleMetadata:
    if isinstance(champion, PolicyBundleMetadata):
        return champion
    if isinstance(champion, TaskChampion):
        champion = champion.selected
    if isinstance(champion, RankedCandidate):
        champion = champion.candidate
    if isinstance(champion, CandidateEvaluation):
        metadata = validate_bundle(champion.bundle_dir)
        if metadata.bundle_digest != champion.bundle_digest:
            raise PoolValidationError("champion bundle digest changed after evaluation")
        return metadata
    if isinstance(champion, (str, Path)):
        return validate_bundle(champion)
    raise TypeError(f"unsupported champion object: {type(champion).__name__}")


def _bound_value(task_spec: Any, name: str) -> str:
    if isinstance(task_spec, Mapping):
        value = task_spec.get(name, "")
    else:
        value = getattr(task_spec, name, "")
    return str(value)


def _opaque_id(
    pool_id: str,
    task_spec_digest: str,
) -> str:
    # Tie breaking is selector-visible.  Bind its token only to the public
    # TaskSpec, never to a source task label, algorithm, or policy payload.
    payload = "\0".join((pool_id, task_spec_digest))
    return "lw-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_pool(
    task_specs: Mapping[str, Any],
    champions: Mapping[str, Any],
    *,
    protocol: FrozenProtocol,
) -> BuiltPool:
    """Build public/private views using values derived only from Γ."""

    config = protocol.config
    try:
        pool_config = config["pool"]
        reducer_config = config["reducer"]
        pool_id = str(pool_config["pool_id"])
        checkpoint_outer = int(pool_config["checkpoint_outer"])
        expected_environment_steps = int(pool_config["actual_environment_steps"])
        expected_entries = len(protocol.env_schemas)
        kernel_bandwidth = float(protocol.packed_layout["kernel_bandwidth"])
        latent_dim = int(protocol.packed_layout["latent_dim"])
        support_budget = int(protocol.packed_layout["support_budget"])
        reconstruction_tolerance = float(
            reducer_config["reconstruction_tolerance"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PoolValidationError("FrozenProtocol lacks pool/reducer bindings") from error
    protocol_id = protocol.protocol_id

    spec_tasks = set(task_specs)
    champion_tasks = set(champions)
    if spec_tasks != champion_tasks:
        raise PoolValidationError(
            f"TaskSpec/champion coverage differs: specs_only={sorted(spec_tasks-champion_tasks)}, "
            f"champions_only={sorted(champion_tasks-spec_tasks)}"
        )
    registered_tasks = set(protocol.env_schemas)
    if spec_tasks != registered_tasks:
        raise PoolValidationError(
            f"pool task coverage differs from FrozenProtocol: "
            f"missing={sorted(registered_tasks-spec_tasks)}, "
            f"unexpected={sorted(spec_tasks-registered_tasks)}"
        )
    if len(spec_tasks) != int(expected_entries):
        raise PoolValidationError(f"expected {expected_entries} source tasks, found {len(spec_tasks)}")

    entries: list[SelectorEntry] = []
    records: list[RegistryRecord] = []
    for task in sorted(spec_tasks):
        bound_task = _bound_value(task_specs[task], "source_task")
        if bound_task != task:
            raise PoolValidationError(
                f"TaskSpec mapping key {task!r} is not bound to source task {bound_task!r}"
            )
        if not _bound_value(task_specs[task], "source_dataset_digest"):
            raise PoolValidationError(f"TaskSpec for {task!r} has no source dataset digest")
        if not _bound_value(task_specs[task], "source_dataset_manifest_digest"):
            raise PoolValidationError(
                f"TaskSpec for {task!r} has no source dataset manifest digest"
            )
        reduction_error = float(getattr(task_specs[task], "reduction_error", float("nan")))
        if not np.isfinite(reduction_error) or reduction_error > reconstruction_tolerance:
            raise PoolValidationError(
                f"TaskSpec for {task!r} violates the registered reconstruction tolerance"
            )
        task_spec = SelectorTaskSpec.from_rkme(
            task_specs[task],
            protocol_id=protocol_id,
            kernel_bandwidth=kernel_bandwidth,
        )
        if task_spec.latent_dim != latent_dim or task_spec.support_budget != support_budget:
            raise PoolValidationError("TaskSpec dimensions violate FrozenProtocol")
        metadata = _metadata(champions[task])
        if metadata.task != task:
            raise PoolValidationError(f"champion for {task!r} belongs to {metadata.task!r}")
        if metadata.outer_iteration != checkpoint_outer:
            raise PoolValidationError("champion outer iteration violates the pool budget")
        if (
            metadata.environment_steps != expected_environment_steps
        ):
            raise PoolValidationError("champion environment steps violate the pool budget")
        env_schema = protocol.env_schemas[task]
        if (
            metadata.observation_dim != env_schema.observation_dim
            or metadata.action_dim != env_schema.action_dim
        ):
            raise PoolValidationError("champion native dimensions violate EnvSchema")
        opaque_id = _opaque_id(pool_id, task_spec.task_spec_digest)
        entries.append(SelectorEntry(opaque_id, protocol_id, task_spec))
        records.append(
            RegistryRecord(
                opaque_id=opaque_id,
                protocol_id=protocol_id,
                policy_bundle=metadata.bundle_dir,
                policy_bundle_digest=metadata.bundle_digest,
                native_observation_dim=metadata.observation_dim,
                native_action_dim=metadata.action_dim,
                source_task=task,
                provenance={
                    "algorithm": metadata.algorithm,
                    "training_seed": metadata.training_seed,
                    "outer_iteration": metadata.outer_iteration,
                    "environment_steps": metadata.environment_steps,
                },
            )
        )

    # Public ordering must not preserve the private source-task traversal.  The
    # opaque id is itself derived only from the public TaskSpec, so sorting by
    # it is deterministic without exposing the builder's task-label order.
    entries = sorted(entries, key=lambda item: item.opaque_id)
    records = sorted(records, key=lambda item: item.opaque_id)
    pool = LearnwarePool(pool_id, protocol_id, float(kernel_bandwidth), tuple(entries))
    pool.validate_expected_size(expected_entries)
    pool_digest = sha256_json(pool.public_manifest())
    registry = DeploymentRegistry(
        tuple(records), pool_id=pool.pool_id, pool_digest=pool_digest
    )
    registry.validate_against(pool)
    return BuiltPool(pool, registry)
