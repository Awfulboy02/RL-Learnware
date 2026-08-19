"""Single fail-closed command line interface for Policy Learnware v0."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gc
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import fmean, pstdev
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactLayout, ArtifactLayoutError
from .config import ConfigError, ProtocolDraft, load_protocol_draft
from .envs.factory import make_env_adapter
from .envs.inspect import inspect_environments, save_inspections
from .envs.mujoco_playground import (
    MUJOCO_PLAYGROUND_DISTRIBUTION_NAMES,
    mujoco_playground_package_version,
)
from .gates import (
    deployment_gate,
    deterministic_ranking,
    nonoverlapping_half_ranges,
    pairwise_order_agreement,
    ranking_gate,
    retrieval_gate,
    unreduced_gate,
    validate_gate_record,
)
from .hashing import canonicalize, sha256_file, sha256_json
from .io import ArtifactExistsError, read_json
from .policy.bundle import BundleValidationError, validate_bundle
from .policy.championize import (
    CandidateEvaluation,
    ChampionizationResult,
    championize,
)
from .policy.inventory import InventoryReport, scan_policy_inventory
from .policy.loader import load_policy
from .policy.evaluate import (
    evaluate_frozen_policy_returns_batched,
    verify_compiled_policy_parity,
)
from .policy.parity import verify_golden_parity
from .pool.builder import build_pool
from .pool.learnware import load_public_pool, save_public_pool
from .pool.registry import load_private_registry, save_private_registry
from .probe.collector import collect_probe_episodes
from .probe.dataset import (
    DatasetManifest,
    EpisodeDataset,
    load_dataset_artifact,
    save_dataset_artifact,
)
from .probe.seed_plan import SeedPlan
from .representation.canonicalizer import PackedEpisodeDataset, TransitionCanonicalizer
from .representation.encoder import (
    EncoderCheckpoint,
    EncoderConfig as RuntimeEncoderConfig,
    TransitionSemanticEncoder,
    train_transition_encoder,
)
from .representation.normalization import NormalizationStats, fit_normalizer
from .rkme.gaussian import GaussianKernel, calibrate_bandwidth
from .rkme.empirical import (
    build_empirical_kme,
    empirical_mmd2,
    episode_balanced_weights,
)
from .evaluation.retrieval_accel import nested_prefix_self_kernel_sums_jax
from .rkme.distance import empirical_to_reduced_distance
from .rkme.reducer import (
    ReducedRKME,
    ReducerConfig as RuntimeReducerConfig,
    reduce_kme,
)
from .schemas import EnvSchema, FrozenProtocol
from .reuse.selector import (
    NearestSpecSelector,
    SelectionResult,
    target_source_cross_terms,
)
from .evaluation.retrieval import RetrievalTrial, summarize_retrieval
from .evaluation.deployment import (
    DeploymentResult,
)
from .evaluation.metrics import summarize_deployments
from .smoke import run_logic_smoke


CLI_SCHEMA = "policy-learnware.cli-result.v0"
ENVIRONMENT_MANIFEST_SCHEMA = "policy-learnware.environment-artifacts.v0"
NORMALIZATION_MANIFEST_SCHEMA = "policy-learnware.normalization-artifact.v0"
ENCODER_MANIFEST_SCHEMA = "policy-learnware.encoder-artifact.v0"
KERNEL_MANIFEST_SCHEMA = "policy-learnware.kernel-artifact.v0"
INVENTORY_SCHEMA = "policy-learnware.policy-inventory.v0"
VERIFICATION_SCHEMA = "policy-learnware.bundle-verification.v0"

RUNTIME_PACKAGE_DISTRIBUTIONS = (
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "numpy",
    "scipy",
    "mujoco",
)

PROBE_SPLITS = (
    "encoder_train",
    "encoder_validation",
    "kernel_calibration",
    "separability_calibration",
    "source_taskspec",
    "target_query",
)

COMMANDS = (
    "validate-config",
    "smoke",
    "inspect-envs",
    "collect-probe",
    "fit-normalizer",
    "train-encoder",
    "calibrate-kernel",
    "diagnose-unreduced",
    "reduce-task-specs",
    "inventory-policies",
    "verify-policy-bundles",
    "championize",
    "build-pool",
    "evaluate-retrieval",
    "evaluate-deployment",
    "build-report",
)

SUPPORTED_COMMANDS = frozenset(
    {
        "validate-config",
        "smoke",
        "inspect-envs",
        "collect-probe",
        "fit-normalizer",
        "train-encoder",
        "calibrate-kernel",
        "diagnose-unreduced",
        "reduce-task-specs",
        "inventory-policies",
        "verify-policy-bundles",
        "championize",
        "build-pool",
        "evaluate-retrieval",
        "evaluate-deployment",
        "build-report",
    }
)

UNAVAILABLE_REASONS = {
}


class CommandFailure(RuntimeError):
    """An execution gate failed without mutating selector state."""


class CommandUnavailable(CommandFailure):
    """A command contract exists, but its production prerequisites do not."""


def _validated_gate(
    payload: Mapping[str, Any], *, expected_name: str
) -> Any:
    try:
        decision = validate_gate_record(payload.get("gate"), expected_name=expected_name)
    except (TypeError, ValueError) as error:
        raise CommandFailure(f"invalid {expected_name} gate artifact: {error}") from error
    if payload.get("gate_passed") is not decision.passed:
        raise CommandFailure(f"inconsistent legacy pass flag for {expected_name}")
    return decision


def _require_gate_passed(
    payload: Mapping[str, Any], *, expected_name: str, artifact: Path
) -> None:
    decision = _validated_gate(payload, expected_name=expected_name)
    if not decision.passed:
        raise CommandFailure(
            f"{expected_name} gate failed; diagnostics retained at {artifact}"
        )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="parent of artifacts/<pool_id>; never inferred from reproduction_root",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print inputs, outputs, hashes, and seeds without writes or GPU work",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="explicit execution alias; execution is already the default",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only complete, digest-verified immutable outputs",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-learnware-v0", description=__doc__)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)
        if command == "inspect-envs":
            subparser.add_argument("--reset-seed", type=int, default=0)
            subparser.add_argument("--no-jit", action="store_true")
        elif command == "collect-probe":
            subparser.add_argument("--split", choices=PROBE_SPLITS, required=True)
            subparser.add_argument(
                "--task",
                action="append",
                dest="tasks",
                help="collect only a registered task; repeatable (default: all tasks)",
            )
            subparser.add_argument(
                "--bank",
                type=int,
                help="one target_query bank (default: all configured banks)",
            )
            subparser.add_argument("--no-jit", action="store_true")
            subparser.add_argument("--no-vectorized", action="store_true")
        elif command in {"inventory-policies", "championize"}:
            subparser.add_argument("--outer", type=int)
            if command == "inventory-policies":
                subparser.add_argument("--runs-root", type=Path)
            else:
                subparser.add_argument(
                    "--devices",
                    help=(
                        "comma-separated CUDA device ids, or 'auto'; the parent "
                        "launches one resumable championization shard per device"
                    ),
                )
                subparser.add_argument(
                    "--shard-index",
                    type=int,
                    help=argparse.SUPPRESS,
                )
                subparser.add_argument(
                    "--shard-count",
                    type=int,
                    help=argparse.SUPPRESS,
                )
                subparser.add_argument(
                    "--merge-only",
                    action="store_true",
                    help=argparse.SUPPRESS,
                )
        elif command == "verify-policy-bundles":
            subparser.add_argument("--atol", type=float)
            subparser.add_argument("--rtol", type=float)
        elif command in {"evaluate-retrieval", "evaluate-deployment"}:
            subparser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
            subparser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    return parser


def _emit(payload: Mapping[str, Any], *, stream: Any | None = None) -> None:
    resolved_stream = sys.stdout if stream is None else stream
    print(
        json.dumps(
            canonicalize(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        file=resolved_stream,
    )


def _selected_tasks(args: argparse.Namespace, config: ProtocolDraft) -> tuple[str, ...]:
    requested = getattr(args, "tasks", None)
    if not requested:
        return config.environment.tasks
    if len(requested) != len(set(requested)):
        raise CommandFailure("--task contains duplicates")
    unknown = sorted(set(requested).difference(config.environment.tasks))
    if unknown:
        raise CommandFailure(f"tasks are not registered by the protocol: {unknown}")
    return tuple(task for task in config.environment.tasks if task in requested)


def _split_episode_count(config: ProtocolDraft, split: str) -> int:
    fields = {
        "encoder_train": "encoder_train_per_task",
        "encoder_validation": "encoder_validation_per_task",
        "kernel_calibration": "kernel_calibration_per_task",
        "separability_calibration": "separability_calibration_per_task",
        "source_taskspec": "source_taskspec_per_task",
        "target_query": "target_query_max_per_task",
    }
    try:
        return int(getattr(config.episodes, fields[split]))
    except KeyError as error:
        raise CommandFailure(f"split {split!r} is not a random-probe dataset") from error


def _banks(args: argparse.Namespace, config: ProtocolDraft, split: str) -> tuple[int | None, ...]:
    requested = getattr(args, "bank", None)
    if split != "target_query":
        if requested is not None:
            raise CommandFailure("--bank is valid only with --split target_query")
        return (None,)
    count = config.episodes.target_query_banks
    if requested is None:
        return tuple(range(count))
    if requested < 0 or requested >= count:
        raise CommandFailure(f"target_query bank must lie in [0, {count})")
    return (requested,)


def _dataset_units(
    args: argparse.Namespace, config: ProtocolDraft, split: str
) -> tuple[tuple[str, int | None, range], ...]:
    count = _split_episode_count(config, split)
    units: list[tuple[str, int | None, range]] = []
    for bank in _banks(args, config, split):
        identifiers = range(count)
        for task in _selected_tasks(args, config):
            units.append((task, bank, identifiers))
    return tuple(units)


def _seed_namespaces(command: str, args: argparse.Namespace) -> tuple[str, ...]:
    mapping = {
        "collect-probe": (getattr(args, "split", "encoder_train"),),
        "fit-normalizer": ("encoder_train",),
        "train-encoder": ("encoder_train", "encoder_validation"),
        "calibrate-kernel": ("kernel_calibration",),
        "diagnose-unreduced": ("separability_calibration",),
        "reduce-task-specs": ("source_taskspec",),
        "championize": ("championization",),
        "evaluate-retrieval": ("target_query",),
        "evaluate-deployment": ("final_return",),
    }
    return mapping.get(command, ())


def _seed_ranges(
    command: str, args: argparse.Namespace, config: ProtocolDraft
) -> dict[str, Any]:
    plan = SeedPlan(config.project_seed)
    records: list[dict[str, Any]] = []
    for namespace in _seed_namespaces(command, args):
        if namespace in PROBE_SPLITS:
            if namespace == getattr(args, "split", None):
                units = _dataset_units(args, config, namespace)
            else:
                proxy = argparse.Namespace(tasks=None, bank=None)
                units = _dataset_units(proxy, config, namespace)
            for task, bank, episode_ids in units:
                task_index = config.environment.tasks.index(task)
                seeds = tuple(
                    plan.episode(
                        namespace,
                        task_index,
                        index,
                        bank_index=0 if bank is None else bank,
                    )
                    for index in episode_ids
                )
                records.append(
                    {
                        "namespace": namespace,
                        "task": task,
                        "bank": bank,
                        "episode_id_range": [episode_ids.start, episode_ids.stop - 1],
                        "episode_count": len(episode_ids),
                        "reset_seed_range": [
                            min(item.reset_seed for item in seeds),
                            max(item.reset_seed for item in seeds),
                        ],
                        "probe_seed_range": [
                            min(item.probe_seed for item in seeds),
                            max(item.probe_seed for item in seeds),
                        ],
                    }
                )
        else:
            count = (
                config.episodes.championization_per_candidate
                if namespace == "championization"
                else config.episodes.final_return_per_task
            )
            for task_index, task in enumerate(config.environment.tasks):
                reset = [
                    plan.derive(namespace, task_index, index, stream="environment_reset")
                    for index in range(count)
                ]
                policy = [
                    plan.derive(namespace, task_index, index, stream="policy_action")
                    for index in range(count)
                ]
                records.append(
                    {
                        "namespace": namespace,
                        "task": task,
                        "episode_id_range": [0, count - 1],
                        "episode_count": count,
                        "reset_seed_range": [min(reset), max(reset)],
                        "policy_seed_range": [min(policy), max(policy)],
                    }
                )
    return {"project_seed": config.project_seed, "records": records}


def _default_runs_root(config: ProtocolDraft) -> Path:
    return Path(config.runtime.reproduction_root).expanduser().resolve() / "runs"


def _jobs_manifest_path(config: ProtocolDraft) -> Path:
    return Path(config.runtime.reproduction_root).expanduser().resolve() / "jobs_manifest.json"


def _expected_job_ids(config: ProtocolDraft) -> tuple[str, ...]:
    return tuple(
        f"full__{task}__{algorithm}__seed{seed}"
        for seed in range(5)
        for task in config.environment.tasks
        for algorithm in ("ppo", "fpo")
    )


def _plan_paths(
    command: str,
    args: argparse.Namespace,
    config: ProtocolDraft,
    layout: ArtifactLayout,
) -> tuple[list[str], list[str]]:
    inputs: list[Path | str] = [args.config.resolve()]
    outputs: list[Path | str] = []

    if command == "smoke":
        outputs = [layout.smoke_report]
    elif command == "inspect-envs":
        inputs.append(Path(config.runtime.fpo_root).expanduser().resolve())
        outputs = [layout.env_schemas, layout.env_golden_io, layout.environment_manifest]
    elif command == "collect-probe":
        inputs.extend([layout.env_schemas, layout.environment_manifest])
        for task, bank, _ in _dataset_units(args, config, args.split):
            outputs.extend(
                [
                    layout.dataset_npz(args.split, task, bank=bank),
                    layout.dataset_manifest(args.split, task, bank=bank),
                ]
            )
    elif command == "fit-normalizer":
        inputs.extend([layout.env_schemas, layout.environment_manifest])
        for task in config.environment.tasks:
            inputs.extend(
                [
                    layout.dataset_npz("encoder_train", task),
                    layout.dataset_manifest("encoder_train", task),
                ]
            )
        outputs = [layout.normalization, layout.normalization_manifest]
    elif command == "train-encoder":
        inputs.extend(
            [
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization,
                layout.normalization_manifest,
            ]
        )
        for split in ("encoder_train", "encoder_validation"):
            for task in config.environment.tasks:
                inputs.extend(
                    [layout.dataset_npz(split, task), layout.dataset_manifest(split, task)]
                )
        outputs = [layout.encoder_checkpoint, layout.encoder_config, layout.encoder_manifest]
    elif command == "calibrate-kernel":
        inputs.extend(
            [
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization,
                layout.normalization_manifest,
                layout.encoder_checkpoint,
                layout.encoder_config,
                layout.encoder_manifest,
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [
                    layout.dataset_npz("kernel_calibration", task),
                    layout.dataset_manifest("kernel_calibration", task),
                    layout.dataset_npz("source_taskspec", task),
                    layout.dataset_manifest("source_taskspec", task),
                ]
            )
        outputs = [
            layout.kernel,
            layout.kernel_manifest,
            layout.frozen_protocol,
            layout.protocol_manifest,
        ]
    elif command == "diagnose-unreduced":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization,
                layout.normalization_manifest,
                layout.encoder_checkpoint,
                layout.encoder_config,
                layout.encoder_manifest,
                layout.kernel,
                layout.kernel_manifest,
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [
                    layout.dataset_npz("separability_calibration", task),
                    layout.dataset_manifest("separability_calibration", task),
                ]
            )
        outputs = [layout.unreduced_diagnostics, layout.mmd_matrix]
    elif command == "reduce-task-specs":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization,
                layout.encoder_checkpoint,
                layout.encoder_config,
                layout.kernel,
                layout.unreduced_diagnostics,
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [
                    layout.dataset_npz("source_taskspec", task),
                    layout.dataset_manifest("source_taskspec", task),
                    layout.dataset_npz("separability_calibration", task),
                    layout.dataset_manifest("separability_calibration", task),
                ]
            )
            outputs.extend(
                [
                    layout.empirical_summary(task),
                    layout.task_rkme(task),
                    layout.task_rkme_manifest(task),
                ]
            )
        outputs.append(layout.reduced_unreduced_ranking)
    elif command == "inventory-policies":
        inputs.extend(
            [
                getattr(args, "runs_root", None) or _default_runs_root(config),
                _jobs_manifest_path(config),
            ]
        )
        outputs = [layout.policy_inventory]
    elif command == "verify-policy-bundles":
        inputs.extend(
            [
                layout.policy_inventory,
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization_manifest,
                layout.encoder_manifest,
                layout.kernel,
                layout.kernel_manifest,
                Path(config.runtime.fpo_root).expanduser().resolve(),
            ]
        )
        outputs = [layout.bundle_verification]
        outputs.extend(layout.parity_report(job_id) for job_id in _expected_job_ids(config))
    elif command == "championize":
        inputs.extend(
            [
                layout.policy_inventory,
                layout.bundle_verification,
                layout.env_schemas,
                layout.environment_manifest,
                Path(config.runtime.fpo_root).expanduser().resolve(),
            ]
        )
        outputs = [
            layout.championization_candidates_dir,
            layout.championization_returns,
            layout.championization,
        ]
    elif command == "build-pool":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.kernel,
                layout.policy_inventory,
                layout.bundle_verification,
                layout.championization_returns,
                layout.championization,
                layout.reduced_unreduced_ranking,
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [layout.task_rkme(task), layout.task_rkme_manifest(task)]
            )
        outputs.extend(
            [layout.selector_pool_dir, layout.private_registry, layout.pool_manifest]
        )
        outputs.extend(layout.learnware_manifest(task) for task in config.environment.tasks)
    elif command == "evaluate-retrieval":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.selector_pool_dir,
                layout.pool_manifest,
                layout.private_registry,
                layout.championization,
                layout.env_schemas,
                layout.environment_manifest,
                layout.normalization,
                layout.encoder_checkpoint,
                layout.encoder_config,
                layout.kernel,
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [layout.task_rkme(task), layout.learnware_manifest(task)]
            )
        for bank in range(config.episodes.target_query_banks):
            for task in config.environment.tasks:
                inputs.extend(
                    [
                        layout.dataset_npz("target_query", task, bank=bank),
                        layout.dataset_manifest("target_query", task, bank=bank),
                    ]
                )
        outputs.extend(
            [layout.retrieval_metrics, layout.retrieval_execution_attestation]
        )
        outputs.extend(
            layout.selection_result(_query_id(task, bank, episode_count))
            for bank in range(config.episodes.target_query_banks)
            for task in config.environment.tasks
            for episode_count in config.episodes.target_query_prefix_grid
        )
    elif command == "evaluate-deployment":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.selector_pool_dir,
                layout.pool_manifest,
                layout.private_registry,
                layout.championization,
                layout.retrieval_metrics,
                layout.retrieval_execution_attestation,
                layout.env_schemas,
                layout.environment_manifest,
                Path(config.runtime.fpo_root).expanduser().resolve(),
            ]
        )
        for task in config.environment.tasks:
            inputs.extend(
                [layout.task_rkme(task), layout.learnware_manifest(task)]
            )
        for bank in range(config.episodes.target_query_banks):
            for task in config.environment.tasks:
                inputs.extend(
                    [
                        layout.dataset_npz("target_query", task, bank=bank),
                        layout.dataset_manifest("target_query", task, bank=bank),
                    ]
                )
        inputs.extend(
            layout.selection_result(_query_id(task, bank, episode_count))
            for bank in range(config.episodes.target_query_banks)
            for task in config.environment.tasks
            for episode_count in config.episodes.target_query_prefix_grid
        )
        outputs.append(layout.deployment_metrics)
        outputs.extend(
            layout.deployment_result(_query_id(task, bank, episode_count))
            for bank in range(config.episodes.target_query_banks)
            for task in config.environment.tasks
            for episode_count in config.episodes.target_query_prefix_grid
        )
    elif command == "build-report":
        inputs.extend(
            [
                layout.frozen_protocol,
                layout.protocol_manifest,
                layout.retrieval_metrics,
                layout.retrieval_execution_attestation,
                layout.deployment_metrics,
                layout.unreduced_diagnostics,
                layout.reduced_unreduced_ranking,
                layout.mmd_matrix,
            ]
        )
        outputs = [layout.summary]

    return [str(path) for path in inputs], [str(path) for path in outputs]


def _dry_run_payload(
    command: str,
    args: argparse.Namespace,
    config: ProtocolDraft,
    layout: ArtifactLayout,
) -> dict[str, Any]:
    inputs, outputs = _plan_paths(command, args, config, layout)
    payload: dict[str, Any] = {
        "schema": CLI_SCHEMA,
        "status": "dry_run",
        "command": command,
        "execution_supported": command in SUPPORTED_COMMANDS,
        "config": str(args.config.resolve()),
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "artifact_pool_root": str(layout.pool_root),
        "inputs": inputs,
        "outputs": outputs,
        "seed_ranges": _seed_ranges(command, args, config),
        "will_write": False,
        "will_execute_gpu_work": False,
        "selector_boundary": {
            "public_selection_key": "opaque_id",
            "policy_bundle_resolution": "private deployment registry after selection",
            "candidate_rollout_during_selection": False,
        },
    }
    if command not in SUPPORTED_COMMANDS:
        payload["fail_closed_reason"] = UNAVAILABLE_REASONS[command]
    return payload


def _validate_command_arguments(args: argparse.Namespace, config: ProtocolDraft) -> None:
    if args.command in {"inventory-policies", "championize"}:
        outer = config.pool.checkpoint_outer if args.outer is None else args.outer
        if outer != config.pool.checkpoint_outer:
            raise CommandFailure(
                f"{args.command} outer {outer} differs from fixed config outer "
                f"{config.pool.checkpoint_outer}"
            )
    if args.command == "inspect-envs" and args.reset_seed < 0:
        raise CommandFailure("--reset-seed must be non-negative")
    if args.command == "verify-policy-bundles":
        atol = config.policy.parity_atol if args.atol is None else args.atol
        rtol = config.policy.parity_rtol if args.rtol is None else args.rtol
        if atol < 0 or rtol < 0:
            raise CommandFailure("parity tolerances must be non-negative")
    if args.command == "championize":
        shard_index = getattr(args, "shard_index", None)
        shard_count = getattr(args, "shard_count", None)
        if (shard_index is None) != (shard_count is None):
            raise CommandFailure("--shard-index and --shard-count must be supplied together")
        if shard_count is not None and (
            shard_count <= 0 or shard_index < 0 or shard_index >= shard_count
        ):
            raise CommandFailure("championization shard index/count are out of range")
        if getattr(args, "devices", None) is not None and shard_count is not None:
            raise CommandFailure("--devices cannot be combined with worker shard arguments")
        if shard_count is not None and os.environ.get(CHAMPIONIZATION_WORKER_ENV) != "1":
            raise CommandFailure(
                "championization shard arguments are reserved for parent-launched workers"
            )
        if getattr(args, "merge_only", False) and (
            getattr(args, "devices", None) is not None or shard_count is not None
        ):
            raise CommandFailure("--merge-only cannot launch or act as a worker shard")
    if args.command in {"evaluate-retrieval", "evaluate-deployment"}:
        shard_index = getattr(args, "shard_index", None)
        shard_count = getattr(args, "shard_count", None)
        if (shard_index is None) != (shard_count is None):
            raise CommandFailure("--shard-index and --shard-count must be supplied together")
        maximum_shards = (
            config.episodes.target_query_banks * len(config.environment.tasks)
            if args.command == "evaluate-retrieval"
            else len(config.environment.tasks) ** 2
        )
        if shard_count is not None and (
            shard_count <= 0
            or shard_count > maximum_shards
            or shard_index < 0
            or shard_index >= shard_count
        ):
            raise CommandFailure(f"{args.command} shard index/count are out of range")
        if shard_count is not None and not args.resume:
            raise CommandFailure(f"{args.command} shard workers require --resume")


def _assert_output_state(
    outputs: Sequence[Path],
    *,
    resume: bool,
    completion_manifest: Path | None = None,
) -> None:
    existing = [path for path in outputs if path.exists()]
    if not existing:
        return
    if not resume:
        raise ArtifactExistsError(
            "refusing to overwrite immutable outputs: " + ", ".join(str(path) for path in existing)
        )
    if completion_manifest is None or not completion_manifest.is_file():
        raise ArtifactLayoutError(
            "resume found partial outputs without a completion manifest: "
            + ", ".join(str(path) for path in existing)
        )


def _manifest_file(layout: ArtifactLayout, path: Path, digest: str | None = None) -> dict[str, str]:
    return {"path": layout.relative(path), "sha256": digest or sha256_file(path)}


def _resume_manifest(
    layout: ArtifactLayout,
    manifest_path: Path,
    *,
    config_hash: str,
) -> Mapping[str, Any] | None:
    if not manifest_path.is_file():
        return None
    manifest = layout.verify_manifest_files(manifest_path)
    if manifest.get("complete") is not True:
        raise ArtifactLayoutError(f"resume manifest is incomplete: {manifest_path}")
    if manifest.get("protocol_draft_hash") != config_hash:
        raise ArtifactLayoutError(f"resume manifest config hash mismatch: {manifest_path}")
    return manifest


def _verify_fpo_checkout(config: ProtocolDraft) -> dict[str, str]:
    root = Path(config.runtime.fpo_root).expanduser().resolve()
    if not root.is_dir():
        raise CommandFailure(f"FPO root does not exist: {root}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise CommandFailure(f"cannot verify FPO checkout ({' '.join(arguments)}): {detail}")
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit != config.runtime.fpo_commit:
        raise CommandFailure(
            f"FPO commit mismatch: expected {config.runtime.fpo_commit}, found {commit}"
        )
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise CommandFailure("FPO checkout has tracked modifications")
    return {"root": str(root), "commit": commit, "tracked_status": "clean"}


def _configure_jax_runtime() -> None:
    """Apply non-networking, non-preallocating defaults before lazy JAX imports."""

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")


def _implementation_source_digest() -> str:
    """Legacy v0 full-source digest retained for old protocol verification."""

    package_root = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(package_root)): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    if not files:
        raise CommandFailure("cannot fingerprint the Policy Learnware source tree")
    return sha256_json(files)


_TASKSPEC_SEMANTIC_SOURCE_FILES = (
    "schemas.py",
    "probe/collector.py",
    "probe/dataset.py",
    "probe/gaussian.py",
    "probe/seed_plan.py",
    "envs/factory.py",
    "envs/mujoco_playground.py",
    "representation/canonicalizer.py",
    "representation/contrastive.py",
    "representation/encoder.py",
    "representation/normalization.py",
    "rkme/empirical.py",
    "rkme/gaussian.py",
    "rkme/distance.py",
    "rkme/reducer.py",
)

# These v0 protocols were frozen before source binding was split into semantic
# and orchestration scopes.  Their full-tree digests were audited at migration
# time against the semantic file set above.  Pure CLI/policy-evaluator changes
# may use them only while that semantic digest remains unchanged.
# The 2026-08-19 Git bootstrap removed one terminal blank line from
# ``envs/factory.py``.  The pre-bootstrap backup hashes to the earlier semantic
# digest and has an identical Python AST; the current digest below records that
# audited, syntax-preserving normalization rather than widening the migration.
_LEGACY_SOURCE_TO_TASKSPEC_SEMANTIC_SOURCE = {
    "982b300d2e978ea77d837a23a40e751b4ad396234e6051463d168c84a817bd61": (
        "65d9ab4406542f8ae78fbcd20d32449f9304d7eed172dc1278ac0a56e78fff16"
    ),
    "a67d1a46f14eefd79317c7f453a63670083d7060576f103908fc949d4221f5da": (
        "65d9ab4406542f8ae78fbcd20d32449f9304d7eed172dc1278ac0a56e78fff16"
    ),
}


def _taskspec_semantic_source_digest() -> str:
    """Fingerprint only code that can change TaskSpec mathematical semantics."""

    package_root = Path(__file__).resolve().parent
    files = {
        relative: sha256_file(package_root / relative)
        for relative in _TASKSPEC_SEMANTIC_SOURCE_FILES
    }
    return sha256_json(files)


def _runtime_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in RUNTIME_PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    versions["playground"] = mujoco_playground_package_version() or "unavailable"
    return versions


def _verify_frozen_protocol_runtime(protocol: FrozenProtocol) -> None:
    """Fail closed when persisted protocol code or package bindings drift."""

    frozen_semantic_source = protocol.component_digests.get(
        "taskspec_semantic_source"
    )
    frozen_legacy_source = protocol.component_digests.get("implementation_source")
    if frozen_semantic_source is None and frozen_legacy_source is None:
        raise CommandFailure(
            "FrozenProtocol lacks the implementation_source/taskspec_semantic_source "
            "runtime binding"
        )
    if frozen_semantic_source is not None:
        current_source = _taskspec_semantic_source_digest()
        if frozen_semantic_source != current_source:
            raise CommandFailure(
                "FrozenProtocol taskspec_semantic_source differs from the "
                "current semantic source tree"
            )
    elif frozen_legacy_source != _implementation_source_digest():
        migrated_semantic_source = _LEGACY_SOURCE_TO_TASKSPEC_SEMANTIC_SOURCE.get(
            str(frozen_legacy_source)
        )
        if (
            migrated_semantic_source is None
            or migrated_semantic_source != _taskspec_semantic_source_digest()
        ):
            raise CommandFailure(
                "FrozenProtocol implementation_source differs from the current "
                "source tree and has no valid semantic-source migration"
            )

    frozen_python = protocol.runtime_versions.get("python")
    if frozen_python is None:
        raise CommandFailure("FrozenProtocol lacks the python runtime binding")
    if frozen_python != sys.version:
        raise CommandFailure(
            f"FrozenProtocol python runtime mismatch: {frozen_python!r} (frozen) "
            f"!= {sys.version!r} (current)"
        )

    current_packages = _runtime_package_versions()
    frozen_packages: dict[str, str] = {}
    missing: list[str] = []
    for package in current_packages:
        aliases = (
            MUJOCO_PLAYGROUND_DISTRIBUTION_NAMES
            if package == "playground"
            else (package,)
        )
        bindings = {
            str(protocol.runtime_versions[alias])
            for alias in aliases
            if alias in protocol.runtime_versions
        }
        if not bindings:
            missing.append(package)
            continue
        if len(bindings) != 1:
            raise CommandFailure(
                f"FrozenProtocol has conflicting runtime aliases for {package}"
            )
        frozen_packages[package] = bindings.pop()
    if missing:
        raise CommandFailure(
            "FrozenProtocol lacks runtime package bindings: " + ", ".join(missing)
        )

    mismatches = {
        package: (frozen_packages[package], current_version)
        for package, current_version in current_packages.items()
        if frozen_packages[package] != current_version
    }
    if mismatches:
        detail = ", ".join(
            f"{package}={frozen!r} (frozen) != {current!r} (current)"
            for package, (frozen, current) in sorted(mismatches.items())
        )
        raise CommandFailure(f"FrozenProtocol runtime package mismatch: {detail}")


def _load_env_schemas(
    layout: ArtifactLayout, config: ProtocolDraft
) -> dict[str, EnvSchema]:
    manifest = layout.verify_manifest_files(layout.environment_manifest)
    if manifest.get("protocol_draft_hash") != config.draft_hash:
        raise CommandFailure("environment artifact config hash mismatch")
    payload = read_json(layout.env_schemas)
    if not isinstance(payload, Mapping):
        raise CommandFailure("env_schemas.json must be an object")
    if set(payload) != set(config.environment.tasks):
        raise CommandFailure("environment artifact task coverage differs from config")
    schemas: dict[str, EnvSchema] = {}
    for task in config.environment.tasks:
        record = payload[task]
        if not isinstance(record, Mapping) or not isinstance(record.get("schema"), Mapping):
            raise CommandFailure(f"invalid environment schema record for {task}")
        schema = EnvSchema.from_dict(record["schema"])
        if schema.task != task:
            raise CommandFailure(f"environment schema task mismatch for {task}")
        if schema.backend != config.environment.backend:
            raise CommandFailure(f"environment schema backend mismatch for {task}")
        schemas[task] = schema
    return schemas


def _load_split(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    split: str,
) -> tuple[dict[str, EpisodeDataset], dict[str, DatasetManifest]]:
    datasets: dict[str, EpisodeDataset] = {}
    manifests: dict[str, DatasetManifest] = {}
    for task in config.environment.tasks:
        dataset, manifest = load_dataset_artifact(
            layout.dataset_npz(split, task), layout.dataset_manifest(split, task)
        )
        if (
            manifest.task != task
            or manifest.split != split
            or manifest.protocol_draft_hash != config.draft_hash
        ):
            raise CommandFailure(f"dataset manifest binding mismatch for {split}/{task}")
        datasets[task] = dataset
        manifests[task] = manifest
    return datasets, manifests


def _load_packed_split(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    split: str,
    *,
    schemas: Mapping[str, EnvSchema] | None = None,
    stats: NormalizationStats | None = None,
) -> tuple[dict[str, PackedEpisodeDataset], dict[str, DatasetManifest]]:
    resolved_schemas = dict(schemas or _load_env_schemas(layout, config))
    resolved_stats = stats or NormalizationStats.load_npz(layout.normalization)
    datasets, manifests = _load_split(layout, config, split)
    canonicalizer = TransitionCanonicalizer(
        stats=resolved_stats,
        max_action_dim=config.environment.max_action_dim,
    )
    packed = {
        task: canonicalizer.pack(datasets[task], resolved_schemas[task])
        for task in config.environment.tasks
    }
    return packed, manifests


def _runtime_encoder_config(config: ProtocolDraft) -> RuntimeEncoderConfig:
    return RuntimeEncoderConfig(
        input_dim=(
            4 * config.environment.max_observation_dim
            + 2 * config.environment.max_action_dim
            + 1
        ),
        hidden_dims=config.encoder.hidden_dims,
        latent_dim=config.encoder.latent_dim,
        activation=config.encoder.activation,
        l2_normalize_output=config.encoder.l2_normalize_output,
        temperature=config.encoder.temperature,
        batch_size=config.encoder.batch_size,
        train_steps=config.encoder.train_steps,
        learning_rate=config.encoder.learning_rate,
        weight_decay=config.encoder.weight_decay,
        validation_interval=config.encoder.validation_interval,
        validation_batches=config.encoder.validation_batches,
        seed=config.encoder.seed,
    )


def _handle_validate_config(
    _args: argparse.Namespace, config: ProtocolDraft, _layout: ArtifactLayout
) -> dict[str, Any]:
    return {
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "tasks": list(config.environment.tasks),
        "checkpoint_outer": config.pool.checkpoint_outer,
        "actual_environment_steps": config.pool.actual_environment_steps,
        "effective_task_balanced_batch_size": config.effective_task_balanced_batch_size,
    }


def _handle_smoke(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    if args.resume and layout.smoke_report.is_file():
        payload = read_json(layout.smoke_report)
        if (
            isinstance(payload, Mapping)
            and payload.get("protocol_draft_hash") == config.draft_hash
            and payload.get("passed") is True
        ):
            return {"resumed": True, "report": str(layout.smoke_report)}
        raise ArtifactLayoutError("existing smoke report is incomplete or belongs to another config")
    _assert_output_state([layout.smoke_report], resume=args.resume)
    result = run_logic_smoke()
    payload = {
        "schema": "policy-learnware.logic-smoke.v0",
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        **result.to_dict(),
    }
    if not result.passed:
        raise CommandFailure("logic smoke returned passed=false")
    digest = layout.publish_json(layout.smoke_report, payload)
    return {"passed": True, "report": str(layout.smoke_report), "sha256": digest}


def _handle_inspect_envs(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [layout.env_schemas, layout.env_golden_io, layout.environment_manifest]
    if args.resume:
        resumed = _resume_manifest(
            layout, layout.environment_manifest, config_hash=config.draft_hash
        )
        if resumed is not None:
            return {"resumed": True, "manifest": str(layout.environment_manifest)}
    _assert_output_state(
        outputs, resume=args.resume, completion_manifest=layout.environment_manifest
    )
    source = _verify_fpo_checkout(config)
    _configure_jax_runtime()
    inspections = inspect_environments(
        config, reset_seed=args.reset_seed, jit=not args.no_jit
    )
    digests = save_inspections(inspections, layout.protocol_dir, overwrite=False)
    manifest = {
        "schema": ENVIRONMENT_MANIFEST_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "reset_seed": args.reset_seed,
        "jit": not args.no_jit,
        "runtime_source": source,
        "tasks": list(config.environment.tasks),
        "files": {
            "env_schemas": _manifest_file(
                layout, layout.env_schemas, digests["env_schemas"]
            ),
            "env_golden_io": _manifest_file(
                layout, layout.env_golden_io, digests["env_golden_io"]
            ),
        },
    }
    manifest_digest = layout.publish_json(layout.environment_manifest, manifest)
    return {
        "tasks": len(inspections),
        "manifest": str(layout.environment_manifest),
        "manifest_sha256": manifest_digest,
        "files": manifest["files"],
    }


def _existing_dataset_seed_records(
    layout: ArtifactLayout,
) -> list[tuple[str, DatasetManifest]]:
    records: list[tuple[str, DatasetManifest]] = []
    if not layout.datasets_dir.is_dir():
        return records
    for path in sorted(layout.datasets_dir.rglob("*.json")):
        payload = read_json(path)
        manifest = DatasetManifest.from_dict(payload)
        owner = str(path.parent.relative_to(layout.datasets_dir))
        records.append((owner, manifest))
    return records


def _assert_seed_disjoint(
    records: Iterable[tuple[str, DatasetManifest]],
) -> None:
    owner_by_seed: dict[tuple[str, int, int], str] = {}
    for owner, manifest in records:
        for reset_seed, probe_seed in zip(
            manifest.reset_seeds, manifest.probe_seeds, strict=True
        ):
            key = (manifest.task, reset_seed, probe_seed)
            previous = owner_by_seed.get(key)
            if previous is not None and previous != owner:
                raise CommandFailure(
                    f"dataset seed overlap between {previous!r} and {owner!r}: {key}"
                )
            owner_by_seed[key] = owner


def _handle_collect_probe(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    schemas = _load_env_schemas(layout, config)
    _configure_jax_runtime()
    units = _dataset_units(args, config, args.split)
    collected: dict[tuple[str, int | None], EpisodeDataset] = {}
    resumed: list[dict[str, Any]] = []

    for task, bank, _ in units:
        npz_path = layout.dataset_npz(args.split, task, bank=bank)
        manifest_path = layout.dataset_manifest(args.split, task, bank=bank)
        present = (npz_path.exists(), manifest_path.exists())
        if any(present):
            if not args.resume or not all(present):
                raise ArtifactExistsError(
                    f"dataset output is complete/partial and immutable: {npz_path}, {manifest_path}"
                )
            dataset, manifest = load_dataset_artifact(npz_path, manifest_path)
            if (
                manifest.protocol_draft_hash != config.draft_hash
                or manifest.task != task
                or manifest.split != args.split
            ):
                raise ArtifactLayoutError(f"resume dataset binding mismatch: {manifest_path}")
            expected_count = _split_episode_count(config, args.split)
            if dataset.episode_count != expected_count:
                raise ArtifactLayoutError(f"resume dataset episode count mismatch: {npz_path}")
            resumed.append(
                {"task": task, "bank": bank, "digest": manifest.dataset_sha256}
            )

    for task in _selected_tasks(args, config):
        task_units = [unit for unit in units if unit[0] == task]
        missing = [
            unit
            for unit in task_units
            if not layout.dataset_npz(args.split, task, bank=unit[1]).exists()
        ]
        if not missing:
            continue
        adapter = make_env_adapter(task, config, jit=not args.no_jit)
        if adapter.schema.digest != schemas[task].digest:
            raise CommandFailure(f"live environment schema drift for {task}")
        for _, bank, episode_ids in missing:
            collected[(task, bank)] = collect_probe_episodes(
                adapter,
                args.split,
                episode_ids,
                config,
                seed_plan=SeedPlan(config.project_seed),
                task_index=config.environment.tasks.index(task),
                bank_index=0 if bank is None else bank,
                prefer_vectorized=not args.no_vectorized,
            )

    records = _existing_dataset_seed_records(layout)
    for (task, bank), dataset in collected.items():
        manifest = DatasetManifest.from_dataset(
            dataset,
            split=args.split,
            task=task,
            protocol_draft_hash=config.draft_hash,
        )
        owner = str(layout.dataset_dir(args.split, bank=bank).relative_to(layout.datasets_dir))
        records.append((owner, manifest))
    _assert_seed_disjoint(records)

    published: list[dict[str, Any]] = []
    for task, bank, _ in units:
        dataset = collected.get((task, bank))
        if dataset is None:
            continue
        manifest = save_dataset_artifact(
            dataset,
            npz_path=layout.dataset_npz(args.split, task, bank=bank),
            manifest_path=layout.dataset_manifest(args.split, task, bank=bank),
            split=args.split,
            task=task,
            protocol_draft_hash=config.draft_hash,
            overwrite=False,
        )
        published.append(
            {
                "task": task,
                "bank": bank,
                "episode_count": manifest.episode_count,
                "transition_count": manifest.transition_count,
                "dataset_sha256": manifest.dataset_sha256,
                "npz": str(layout.dataset_npz(args.split, task, bank=bank)),
                "manifest": str(layout.dataset_manifest(args.split, task, bank=bank)),
            }
        )
    return {"split": args.split, "published": published, "resumed": resumed}


def _handle_fit_normalizer(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [layout.normalization, layout.normalization_manifest]
    if args.resume:
        resumed = _resume_manifest(
            layout, layout.normalization_manifest, config_hash=config.draft_hash
        )
        if resumed is not None:
            NormalizationStats.load_npz(layout.normalization)
            return {"resumed": True, "manifest": str(layout.normalization_manifest)}
    _assert_output_state(
        outputs, resume=args.resume, completion_manifest=layout.normalization_manifest
    )
    schemas = _load_env_schemas(layout, config)
    datasets, dataset_manifests = _load_split(layout, config, "encoder_train")
    stats = fit_normalizer(
        datasets,
        schemas,
        max_observation_dim=config.environment.max_observation_dim,
        std_floor=config.normalization.std_floor,
        role="source",
        include_next_observation=config.normalization.include_next_observation,
    )
    normalization_digest = stats.save_npz(layout.normalization, overwrite=False)
    manifest = {
        "schema": NORMALIZATION_MANIFEST_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "fit_split": "encoder_train",
        "source_dataset_digests": {
            task: dataset_manifests[task].dataset_sha256 for task in config.environment.tasks
        },
        "source_task_count": stats.source_task_count,
        "max_observation_dim": stats.max_observation_dim,
        "reward_mean": stats.reward_mean,
        "reward_std": stats.reward_std,
        "std_floor": stats.std_floor,
        "files": {
            "normalization": _manifest_file(
                layout, layout.normalization, normalization_digest
            )
        },
    }
    manifest_digest = layout.publish_json(layout.normalization_manifest, manifest)
    return {
        "manifest": str(layout.normalization_manifest),
        "manifest_sha256": manifest_digest,
        "normalization_sha256": normalization_digest,
    }


def _handle_train_encoder(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [layout.encoder_checkpoint, layout.encoder_config, layout.encoder_manifest]
    if args.resume:
        resumed = _resume_manifest(layout, layout.encoder_manifest, config_hash=config.draft_hash)
        if resumed is not None:
            _configure_jax_runtime()
            EncoderCheckpoint.load(layout.encoder_checkpoint, read_json(layout.encoder_config))
            return {"resumed": True, "manifest": str(layout.encoder_manifest)}
    _assert_output_state(outputs, resume=args.resume, completion_manifest=layout.encoder_manifest)
    schemas = _load_env_schemas(layout, config)
    normalization_manifest = _resume_manifest(
        layout, layout.normalization_manifest, config_hash=config.draft_hash
    )
    if normalization_manifest is None:
        raise CommandFailure("normalization completion manifest is missing")
    stats = NormalizationStats.load_npz(layout.normalization)
    train, train_manifests = _load_packed_split(
        layout, config, "encoder_train", schemas=schemas, stats=stats
    )
    validation, validation_manifests = _load_packed_split(
        layout, config, "encoder_validation", schemas=schemas, stats=stats
    )
    runtime_config = _runtime_encoder_config(config)
    if {dataset.packed_dim for dataset in train.values()} != {runtime_config.input_dim}:
        raise CommandFailure("canonical packed width differs from encoder input_dim")
    _configure_jax_runtime()
    checkpoint = train_transition_encoder(train, validation, runtime_config)
    digests = checkpoint.save(
        layout.encoder_checkpoint, layout.encoder_config, overwrite=False
    )
    manifest = {
        "schema": ENCODER_MANIFEST_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "checkpoint_metric": config.encoder.checkpoint_metric,
        "checkpoint_tie_break": config.encoder.checkpoint_tie_break,
        "best_step": checkpoint.best_step,
        "best_validation_loss": checkpoint.best_validation_loss,
        "training_history": list(checkpoint.training_history),
        "source_dataset_digests": {
            "encoder_train": {
                task: train_manifests[task].dataset_sha256 for task in config.environment.tasks
            },
            "encoder_validation": {
                task: validation_manifests[task].dataset_sha256
                for task in config.environment.tasks
            },
        },
        "normalization_sha256": sha256_file(layout.normalization),
        "files": {
            "checkpoint": _manifest_file(layout, layout.encoder_checkpoint, digests["checkpoint"]),
            "config": _manifest_file(layout, layout.encoder_config, digests["config"]),
        },
    }
    manifest_digest = layout.publish_json(layout.encoder_manifest, manifest)
    return {
        "manifest": str(layout.encoder_manifest),
        "manifest_sha256": manifest_digest,
        "best_step": checkpoint.best_step,
        "best_validation_loss": checkpoint.best_validation_loss,
    }


def _handle_calibrate_kernel(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [
        layout.kernel,
        layout.kernel_manifest,
        layout.frozen_protocol,
        layout.protocol_manifest,
    ]
    if args.resume:
        resumed = _resume_manifest(
            layout, layout.protocol_manifest, config_hash=config.draft_hash
        )
        if resumed is not None:
            _load_frozen_protocol(layout, config)
            kernel = GaussianKernel.load_json(layout.kernel)
            return {
                "resumed": True,
                "manifest": str(layout.kernel_manifest),
                "bandwidth": kernel.bandwidth,
            }
    _assert_output_state(
        outputs, resume=args.resume, completion_manifest=layout.protocol_manifest
    )
    encoder_manifest = _resume_manifest(
        layout, layout.encoder_manifest, config_hash=config.draft_hash
    )
    if encoder_manifest is None:
        raise CommandFailure("encoder completion manifest is missing")
    normalization_manifest = _resume_manifest(
        layout, layout.normalization_manifest, config_hash=config.draft_hash
    )
    if normalization_manifest is None:
        raise CommandFailure("normalization completion manifest is missing")
    if encoder_manifest.get("normalization_sha256") != sha256_file(layout.normalization):
        raise CommandFailure("normalization artifact changed after encoder training")
    schemas = _load_env_schemas(layout, config)
    stats = NormalizationStats.load_npz(layout.normalization)
    packed, dataset_manifests = _load_packed_split(
        layout, config, "kernel_calibration", schemas=schemas, stats=stats
    )
    _configure_jax_runtime()
    checkpoint = EncoderCheckpoint.load(
        layout.encoder_checkpoint, read_json(layout.encoder_config)
    )
    encoder = TransitionSemanticEncoder(checkpoint)
    semantic_events = {
        task: SimpleNamespace(
            points=encoder.encode(packed[task].packed),
            episode_offsets=packed[task].episode_offsets,
        )
        for task in config.environment.tasks
    }
    bandwidth = calibrate_bandwidth(
        semantic_events,
        calibration_pairs=config.kernel.calibration_pairs,
        seed=config.kernel.seed,
    )
    kernel = GaussianKernel(bandwidth)
    kernel_digest = kernel.save_json(layout.kernel, overwrite=False)
    manifest = {
        "schema": KERNEL_MANIFEST_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "type": "gaussian",
        "bandwidth": bandwidth,
        "calibration_pairs": config.kernel.calibration_pairs,
        "seed": config.kernel.seed,
        "pair_sampling": config.kernel.pair_sampling,
        "source_dataset_digests": {
            task: dataset_manifests[task].dataset_sha256 for task in config.environment.tasks
        },
        "encoder_sha256": sha256_file(layout.encoder_checkpoint),
        "files": {"kernel": _manifest_file(layout, layout.kernel, kernel_digest)},
    }
    manifest_digest = layout.publish_json(layout.kernel_manifest, manifest)
    env_schemas = _load_env_schemas(layout, config)
    source_datasets, source_manifests = _load_split(
        layout, config, "source_taskspec"
    )
    del source_datasets
    source_manifest_digests = {
        task: sha256_file(layout.dataset_manifest("source_taskspec", task))
        for task in config.environment.tasks
    }
    probe_digest = sha256_json(
        {
            "collector": sha256_file(Path(__file__).parent / "probe" / "collector.py"),
            "gaussian": sha256_file(Path(__file__).parent / "probe" / "gaussian.py"),
            "environment_collector": sha256_file(
                Path(__file__).parent / "envs" / "mujoco_playground.py"
            ),
            "rng_backend": config.probe.rng_backend,
        }
    )
    frozen = FrozenProtocol.create(
        config=config.to_dict(),
        env_schemas=env_schemas,
        packed_layout={
            "width": 4 * config.environment.max_observation_dim
            + 2 * config.environment.max_action_dim
            + 1,
            "max_observation_dim": config.environment.max_observation_dim,
            "max_action_dim": config.environment.max_action_dim,
            "latent_dim": config.encoder.latent_dim,
            "support_budget": config.reducer.support_budget,
            "kernel_bandwidth": bandwidth,
            "layout_version": "pack109-padding-mask-v0",
        },
        component_digests={
            "environment_manifest": sha256_file(layout.environment_manifest),
            "probe_implementation": probe_digest,
            "normalization": sha256_file(layout.normalization_manifest),
            "encoder": sha256_file(layout.encoder_manifest),
            "kernel": sha256_file(layout.kernel_manifest),
            "source_dataset_manifests": sha256_json(source_manifest_digests),
            "taskspec_semantic_source": _taskspec_semantic_source_digest(),
        },
        runtime_versions={
            "python": sys.version,
            "fpo_commit": config.runtime.fpo_commit,
            "environment_backend": config.environment.backend,
            "probe_rng_backend": config.probe.rng_backend,
            **_runtime_package_versions(),
        },
    )
    protocol_digest = frozen.save(layout.frozen_protocol, overwrite=False)
    protocol_manifest = {
        "schema": "policy-learnware.protocol-artifacts.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": frozen.protocol_id,
        "pool_id": config.pool.pool_id,
        "source_dataset_manifest_digests": source_manifest_digests,
        "files": {
            "protocol": _manifest_file(
                layout, layout.frozen_protocol, protocol_digest
            ),
            "environment_manifest": _manifest_file(
                layout, layout.environment_manifest
            ),
            "normalization_manifest": _manifest_file(
                layout, layout.normalization_manifest
            ),
            "encoder_manifest": _manifest_file(layout, layout.encoder_manifest),
            "kernel_manifest": _manifest_file(
                layout, layout.kernel_manifest, manifest_digest
            ),
        },
    }
    protocol_manifest_digest = layout.publish_json(
        layout.protocol_manifest, protocol_manifest
    )
    return {
        "manifest": str(layout.protocol_manifest),
        "manifest_sha256": protocol_manifest_digest,
        "bandwidth": bandwidth,
        "protocol_id": frozen.protocol_id,
    }


def _load_frozen_protocol(
    layout: ArtifactLayout, config: ProtocolDraft
) -> FrozenProtocol:
    manifest = layout.verify_manifest_files(layout.protocol_manifest)
    protocol = FrozenProtocol.load(layout.frozen_protocol)
    if manifest.get("protocol_id") != protocol.protocol_id:
        raise CommandFailure("protocol completion manifest id mismatch")
    if sha256_json(protocol.config) != config.draft_hash:
        raise CommandFailure("FrozenProtocol config differs from requested draft")
    _verify_frozen_protocol_runtime(protocol)
    kernel = GaussianKernel.load_json(layout.kernel)
    if not np.isclose(
        kernel.bandwidth,
        float(protocol.packed_layout["kernel_bandwidth"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise CommandFailure("FrozenProtocol and kernel bandwidth differ")
    return protocol


def _load_semantic_split(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    split: str,
) -> tuple[dict[str, SimpleNamespace], dict[str, DatasetManifest]]:
    schemas = _load_env_schemas(layout, config)
    stats = NormalizationStats.load_npz(layout.normalization)
    packed, manifests = _load_packed_split(
        layout, config, split, schemas=schemas, stats=stats
    )
    checkpoint = EncoderCheckpoint.load(
        layout.encoder_checkpoint, read_json(layout.encoder_config)
    )
    encoder = TransitionSemanticEncoder(checkpoint)
    semantic = {
        task: SimpleNamespace(
            points=encoder.encode(packed[task].packed),
            episode_offsets=packed[task].episode_offsets,
        )
        for task in config.environment.tasks
    }
    return semantic, manifests


def _episode_subset(
    semantic: SimpleNamespace, start_episode: int, stop_episode: int
) -> SimpleNamespace:
    offsets = np.asarray(semantic.episode_offsets, dtype=np.int64)
    if not 0 <= start_episode < stop_episode <= offsets.size - 1:
        raise CommandFailure("invalid semantic episode subset")
    start = int(offsets[start_episode])
    stop = int(offsets[stop_episode])
    return SimpleNamespace(
        points=np.asarray(semantic.points)[start:stop],
        episode_offsets=offsets[start_episode : stop_episode + 1] - start,
    )


def _validate_unreduced_split_audit(
    payload: Mapping[str, Any], config: ProtocolDraft, layout: ArtifactLayout
) -> None:
    split_protocol = payload.get("split_protocol")
    if not isinstance(split_protocol, Mapping):
        raise CommandFailure("unreduced split audit is missing")
    ranges = split_protocol.get("episode_ranges")
    if (
        split_protocol.get("source_role") != "first_half"
        or split_protocol.get("query_role") != "second_half"
        or split_protocol.get("candidate_sources_use_query_episodes") is not False
        or not isinstance(ranges, Mapping)
        or set(ranges) != set(config.environment.tasks)
    ):
        raise CommandFailure("unreduced split roles/coverage are invalid")
    source_range, query_range = nonoverlapping_half_ranges(
        config.episodes.separability_calibration_per_task
    )
    for task in config.environment.tasks:
        record = ranges[task]
        manifest_payload = read_json(
            layout.dataset_manifest("separability_calibration", task)
        )
        if not isinstance(record, Mapping) or not isinstance(
            manifest_payload, Mapping
        ):
            raise CommandFailure(f"unreduced split audit is invalid for {task}")
        if (
            list(record.get("source_episode_range_half_open", []))
            != list(source_range)
            or list(record.get("query_episode_range_half_open", []))
            != list(query_range)
            or int(record.get("overlap_episode_count", -1)) != 0
            or record.get("dataset_sha256")
            != manifest_payload.get("dataset_sha256")
        ):
            raise CommandFailure(f"unreduced source/query overlap audit failed for {task}")


def _task_float_mapping(
    value: Any, tasks: Sequence[str], *, label: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(tasks):
        raise CommandFailure(f"{label} task coverage is invalid")
    parsed = {task: float(value[task]) for task in tasks}
    if any(not np.isfinite(item) or item < 0.0 for item in parsed.values()):
        raise CommandFailure(f"{label} contains an invalid distance")
    return parsed


def _load_audited_mmd_matrix(
    layout: ArtifactLayout, tasks: Sequence[str]
) -> np.ndarray:
    try:
        lines = layout.mmd_matrix.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(tasks) + 1:
            raise ValueError("row count")
        if lines[0].split(",") != ["", *tasks]:
            raise ValueError("header")
        rows: list[list[float]] = []
        for task, line in zip(tasks, lines[1:], strict=True):
            fields = line.split(",")
            if fields[0] != task or len(fields) != len(tasks) + 1:
                raise ValueError("row label/width")
            rows.append([float(item) for item in fields[1:]])
        matrix = np.asarray(rows, dtype=np.float64)
    except (OSError, TypeError, ValueError) as error:
        raise CommandFailure("unreduced MMD matrix is malformed") from error
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-15)
        or not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-15)
    ):
        raise CommandFailure("unreduced MMD matrix invariants failed")
    return matrix


def _validate_unreduced_diagnostics_artifact(
    payload: Mapping[str, Any],
    config: ProtocolDraft,
    layout: ArtifactLayout,
    *,
    expected_protocol_id: str,
) -> None:
    tasks = config.environment.tasks
    if (
        payload.get("schema") != "policy-learnware.unreduced-diagnostics.v0"
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("protocol_id") != expected_protocol_id
        or payload.get("mmd_matrix_sha256") != sha256_file(layout.mmd_matrix)
    ):
        raise CommandFailure("unreduced diagnostics artifact binding failed")
    _validate_unreduced_split_audit(payload, config, layout)
    matrix = _load_audited_mmd_matrix(layout, tasks)
    within = _task_float_mapping(
        payload.get("within_task_mmd"), tasks, label="within_task_mmd"
    )
    raw_distances = payload.get("split_retrieval_distances")
    raw_selected = payload.get("split_retrieval")
    if (
        not isinstance(raw_distances, Mapping)
        or set(raw_distances) != set(tasks)
        or not isinstance(raw_selected, Mapping)
        or set(raw_selected) != set(tasks)
    ):
        raise CommandFailure("unreduced split retrieval coverage is invalid")
    selected: dict[str, str] = {}
    for task in tasks:
        distances = _task_float_mapping(
            raw_distances[task], tasks, label=f"split distances for {task}"
        )
        expected_selected = deterministic_ranking(distances)[0]
        selected[task] = str(raw_selected[task])
        if selected[task] != expected_selected or not np.isclose(
            distances[task], within[task], rtol=1.0e-10, atol=1.0e-12
        ):
            raise CommandFailure(f"unreduced split retrieval record mismatch for {task}")
    off_diagonal = matrix[~np.eye(len(tasks), dtype=np.bool_)]
    minimum_between = float(np.min(off_diagonal))
    maximum_within = float(max(within.values()))
    split_accuracy = sum(selected[task] == task for task in tasks) / len(tasks)
    if (
        float(payload.get("minimum_between_mmd", float("nan")))
        != minimum_between
        or float(payload.get("maximum_within_mmd", float("nan")))
        != maximum_within
        or float(payload.get("split_retrieval_accuracy", float("nan")))
        != split_accuracy
    ):
        raise CommandFailure("unreduced top-level metrics do not match raw records")
    expected_gate = unreduced_gate(
        minimum_between_mmd=minimum_between,
        maximum_within_mmd=maximum_within,
        split_retrieval_accuracy=split_accuracy,
        minimum_between_within_ratio=(
            config.gates.unreduced.minimum_between_within_ratio
        ),
        minimum_absolute_margin=config.gates.unreduced.minimum_absolute_margin,
        minimum_split_retrieval_accuracy=(
            config.gates.unreduced.minimum_split_retrieval_accuracy
        ),
    )
    if (
        canonicalize(payload.get("gate"))
        != canonicalize(expected_gate.to_dict())
        or payload.get("gate_passed") is not expected_gate.passed
    ):
        raise CommandFailure("unreduced gate is not bound to metrics/config")
    _require_gate_passed(
        payload,
        expected_name="unreduced_separability",
        artifact=layout.unreduced_diagnostics,
    )


def _validate_ranking_artifact(
    payload: Mapping[str, Any],
    config: ProtocolDraft,
    layout: ArtifactLayout,
    *,
    expected_protocol_id: str,
) -> None:
    source_hashes = payload.get("source_dataset_manifest_sha256")
    query_hashes = payload.get("query_dataset_manifest_sha256")
    task_hashes = payload.get("task_rkme_sha256")
    expected_tasks = set(config.environment.tasks)
    if (
        payload.get("schema") != "policy-learnware.reduced-unreduced-ranking.v0"
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("protocol_id") != expected_protocol_id
        or payload.get("source_split") != "source_taskspec"
        or payload.get("query_split") != "separability_calibration"
        or payload.get("source_query_splits_are_distinct") is not True
        or not isinstance(source_hashes, Mapping)
        or not isinstance(query_hashes, Mapping)
        or not isinstance(task_hashes, Mapping)
        or set(source_hashes) != expected_tasks
        or set(query_hashes) != expected_tasks
        or set(task_hashes) != expected_tasks
        or any(
            source_hashes[task]
            != sha256_file(layout.dataset_manifest("source_taskspec", task))
            or query_hashes[task]
            != sha256_file(
                layout.dataset_manifest("separability_calibration", task)
            )
            or task_hashes[task] != sha256_file(layout.task_rkme(task))
            for task in config.environment.tasks
        )
    ):
        raise CommandFailure("reduced/unreduced ranking artifact binding failed")
    queries = payload.get("queries")
    if not isinstance(queries, Mapping) or set(queries) != expected_tasks:
        raise CommandFailure("ranking query coverage is invalid")
    top1_agreements = 0
    pairwise_agreements: list[float] = []
    for query_task in config.environment.tasks:
        record = queries[query_task]
        if not isinstance(record, Mapping):
            raise CommandFailure(f"ranking query is invalid for {query_task}")
        unreduced_distances = _task_float_mapping(
            record.get("unreduced_distances"),
            config.environment.tasks,
            label=f"unreduced ranking distances for {query_task}",
        )
        reduced_distances = _task_float_mapping(
            record.get("reduced_distances"),
            config.environment.tasks,
            label=f"reduced ranking distances for {query_task}",
        )
        unreduced_ranking = deterministic_ranking(unreduced_distances)
        reduced_ranking = deterministic_ranking(reduced_distances)
        top1_agrees = unreduced_ranking[0] == reduced_ranking[0]
        pairwise = pairwise_order_agreement(unreduced_ranking, reduced_ranking)
        query_manifest = read_json(
            layout.dataset_manifest("separability_calibration", query_task)
        )
        if (
            not isinstance(query_manifest, Mapping)
            or record.get("query_dataset_sha256")
            != query_manifest.get("dataset_sha256")
            or list(record.get("unreduced_ranking", []))
            != list(unreduced_ranking)
            or list(record.get("reduced_ranking", [])) != list(reduced_ranking)
            or record.get("unreduced_top1") != unreduced_ranking[0]
            or record.get("reduced_top1") != reduced_ranking[0]
            or record.get("top1_agrees") is not top1_agrees
            or not np.isclose(
                float(record.get("pairwise_order_agreement", float("nan"))),
                pairwise,
                rtol=0.0,
                atol=1.0e-15,
            )
        ):
            raise CommandFailure(f"ranking query record mismatch for {query_task}")
        top1_agreements += int(top1_agrees)
        pairwise_agreements.append(pairwise)
    top1_agreement = top1_agreements / len(config.environment.tasks)
    mean_pairwise = float(np.mean(pairwise_agreements))
    if (
        float(payload.get("top1_agreement", float("nan"))) != top1_agreement
        or not np.isclose(
            float(payload.get("mean_pairwise_order_agreement", float("nan"))),
            mean_pairwise,
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise CommandFailure("ranking top-level metrics do not match query records")
    expected_gate = ranking_gate(
        top1_agreement=top1_agreement,
        minimum_top1_agreement=(
            config.gates.reduced_unreduced_ranking.minimum_top1_agreement
        ),
    )
    if (
        canonicalize(payload.get("gate"))
        != canonicalize(expected_gate.to_dict())
        or payload.get("gate_passed") is not expected_gate.passed
    ):
        raise CommandFailure("ranking gate is not bound to queries/config")
    _require_gate_passed(
        payload,
        expected_name="reduced_unreduced_ranking",
        artifact=layout.reduced_unreduced_ranking,
    )


def _handle_diagnose_unreduced(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [layout.mmd_matrix, layout.unreduced_diagnostics]
    if args.resume and layout.unreduced_diagnostics.is_file():
        payload = read_json(layout.unreduced_diagnostics)
        if (
            isinstance(payload, Mapping)
            and payload.get("complete") is True
            and payload.get("protocol_draft_hash") == config.draft_hash
            and sha256_file(layout.mmd_matrix) == payload.get("mmd_matrix_sha256")
        ):
            protocol = _load_frozen_protocol(layout, config)
            _validate_unreduced_diagnostics_artifact(
                payload,
                config,
                layout,
                expected_protocol_id=protocol.protocol_id,
            )
            return {"resumed": True, "diagnostics": str(layout.unreduced_diagnostics)}
        raise ArtifactLayoutError("unreduced diagnostic resume validation failed")
    _assert_output_state(outputs, resume=args.resume)
    protocol = _load_frozen_protocol(layout, config)
    kernel = GaussianKernel.load_json(layout.kernel)
    semantic, manifests = _load_semantic_split(
        layout, config, "separability_calibration"
    )
    source_halves: dict[str, Any] = {}
    query_halves: dict[str, Any] = {}
    episode_ranges: dict[str, Any] = {}
    for task in config.environment.tasks:
        episodes = int(np.asarray(semantic[task].episode_offsets).size - 1)
        if episodes < 2:
            raise CommandFailure(
                "separability_calibration needs at least two episodes per task"
            )
        source_tuple, query_tuple = nonoverlapping_half_ranges(episodes)
        source_range = list(source_tuple)
        query_range = list(query_tuple)
        source_halves[task] = build_empirical_kme(
            _episode_subset(semantic[task], *source_range),
            kernel,
            protocol_id=protocol.protocol_id,
            dataset_digest=sha256_json(
                {
                    "dataset_sha256": manifests[task].dataset_sha256,
                    "role": "unreduced_source_half",
                    "episode_range": source_range,
                }
            ),
            computation_backend=config.reducer.optimizer_backend,
        )
        query_halves[task] = build_empirical_kme(
            _episode_subset(semantic[task], *query_range),
            kernel,
            protocol_id=protocol.protocol_id,
            dataset_digest=sha256_json(
                {
                    "dataset_sha256": manifests[task].dataset_sha256,
                    "role": "unreduced_query_half",
                    "episode_range": query_range,
                }
            ),
            computation_backend=config.reducer.optimizer_backend,
        )
        episode_ranges[task] = {
            "dataset_sha256": manifests[task].dataset_sha256,
            "source_episode_range_half_open": source_range,
            "query_episode_range_half_open": query_range,
            "overlap_episode_count": 0,
        }
    task_count = len(config.environment.tasks)
    matrix = np.zeros((task_count, task_count), dtype=np.float64)
    for left_index, left_task in enumerate(config.environment.tasks):
        for right_index in range(left_index + 1, task_count):
            right_task = config.environment.tasks[right_index]
            distance = float(
                np.sqrt(
                    empirical_mmd2(
                        source_halves[left_task],
                        source_halves[right_task],
                        computation_backend=config.reducer.optimizer_backend,
                    )
                )
            )
            matrix[left_index, right_index] = distance
            matrix[right_index, left_index] = distance
    within: dict[str, float] = {}
    split_retrieval: dict[str, str] = {}
    split_retrieval_distances: dict[str, dict[str, float]] = {}
    for task in config.environment.tasks:
        within[task] = float(
            np.sqrt(
                empirical_mmd2(
                    source_halves[task],
                    query_halves[task],
                    computation_backend=config.reducer.optimizer_backend,
                )
            )
        )
        query_distances = {
            # MMD is symmetric, but independent GPU reductions of (source, query)
            # and (query, source) can differ by a few float64 ulps.  Reuse the
            # already-audited within-task value so the emitted record is
            # internally consistent without changing the mathematical result.
            candidate: (
                within[task]
                if candidate == task
                else float(
                    np.sqrt(
                        empirical_mmd2(
                            query_halves[task],
                            source_halves[candidate],
                            computation_backend=config.reducer.optimizer_backend,
                        )
                    )
                )
            )
            for candidate in config.environment.tasks
        }
        split_retrieval_distances[task] = query_distances
        split_retrieval[task] = min(
            query_distances, key=lambda candidate: (query_distances[candidate], candidate)
        )
    off_diagonal = matrix[~np.eye(task_count, dtype=np.bool_)]
    minimum_between = float(np.min(off_diagonal))
    maximum_within = float(max(within.values()))
    split_accuracy = sum(
        split_retrieval[task] == task for task in config.environment.tasks
    ) / task_count
    gate = unreduced_gate(
        minimum_between_mmd=minimum_between,
        maximum_within_mmd=maximum_within,
        split_retrieval_accuracy=split_accuracy,
        minimum_between_within_ratio=(
            config.gates.unreduced.minimum_between_within_ratio
        ),
        minimum_absolute_margin=config.gates.unreduced.minimum_absolute_margin,
        minimum_split_retrieval_accuracy=(
            config.gates.unreduced.minimum_split_retrieval_accuracy
        ),
    )
    header = "," + ",".join(config.environment.tasks)
    rows = [header]
    for index, task in enumerate(config.environment.tasks):
        rows.append(
            task + "," + ",".join(f"{value:.17g}" for value in matrix[index])
        )
    csv_digest = layout.publish_text(layout.mmd_matrix, "\n".join(rows) + "\n")
    diagnostics = {
        "schema": "policy-learnware.unreduced-diagnostics.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "gate": gate.to_dict(),
        "gate_passed": gate.passed,
        "split_protocol": {
            "source_role": "first_half",
            "query_role": "second_half",
            "candidate_sources_use_query_episodes": False,
            "episode_ranges": episode_ranges,
        },
        "minimum_between_mmd": minimum_between,
        "maximum_within_mmd": maximum_within,
        "within_task_mmd": within,
        "split_retrieval_accuracy": split_accuracy,
        "split_retrieval": split_retrieval,
        "split_retrieval_distances": split_retrieval_distances,
        "mmd_matrix_sha256": csv_digest,
    }
    digest = layout.publish_json(layout.unreduced_diagnostics, diagnostics)
    _validate_unreduced_diagnostics_artifact(
        diagnostics,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    return {
        "diagnostics": str(layout.unreduced_diagnostics),
        "sha256": digest,
        "gate_passed": gate.passed,
        "minimum_between_mmd": minimum_between,
        "maximum_within_mmd": maximum_within,
    }


def _handle_reduce_task_specs(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [
        path
        for task in config.environment.tasks
        for path in (
            layout.empirical_summary(task),
            layout.task_rkme(task),
            layout.task_rkme_manifest(task),
        )
    ] + [layout.reduced_unreduced_ranking]
    if args.resume and all(path.is_file() for path in outputs):
        protocol = _load_frozen_protocol(layout, config)
        for task in config.environment.tasks:
            manifest = read_json(layout.task_rkme_manifest(task))
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("complete") is not True
                or manifest.get("protocol_draft_hash") != config.draft_hash
                or manifest.get("protocol_id") != protocol.protocol_id
                or sha256_file(layout.task_rkme(task)) != manifest.get("rkme_sha256")
            ):
                raise ArtifactLayoutError(f"TaskSpec resume validation failed for {task}")
        ranking_payload = read_json(layout.reduced_unreduced_ranking)
        if not isinstance(ranking_payload, Mapping):
            raise ArtifactLayoutError("reduced/unreduced ranking resume validation failed")
        _validate_ranking_artifact(
            ranking_payload,
            config,
            layout,
            expected_protocol_id=protocol.protocol_id,
        )
        return {"resumed": True, "task_count": len(config.environment.tasks)}
    _assert_output_state(outputs, resume=args.resume)
    diagnostics = read_json(layout.unreduced_diagnostics)
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("complete") is not True
        or diagnostics.get("protocol_draft_hash") != config.draft_hash
        or diagnostics.get("mmd_matrix_sha256") != sha256_file(layout.mmd_matrix)
    ):
        raise CommandFailure("unreduced separability artifact binding failed")
    protocol = _load_frozen_protocol(layout, config)
    _validate_unreduced_diagnostics_artifact(
        diagnostics,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    kernel = GaussianKernel.load_json(layout.kernel)
    semantic, manifests = _load_semantic_split(layout, config, "source_taskspec")
    reducer_config = RuntimeReducerConfig(
        support_budget=config.reducer.support_budget,
        init=config.reducer.init,
        support_steps=config.reducer.support_steps,
        learning_rate=config.reducer.learning_rate,
        pinv_rcond=config.reducer.pinv_rcond,
        ridge=config.reducer.ridge,
        kmeans_steps=config.reducer.kmeans_steps,
        negative_tolerance=config.reducer.negative_tolerance,
        optimizer_backend=config.reducer.optimizer_backend,
    )
    _configure_jax_runtime()
    published: dict[str, Any] = {}
    empirical_sources: dict[str, Any] = {}
    reduced_sources: dict[str, ReducedRKME] = {}
    for task in config.environment.tasks:
        manifest_path = layout.dataset_manifest("source_taskspec", task)
        manifest_digest = sha256_file(manifest_path)
        empirical = build_empirical_kme(
            semantic[task],
            kernel,
            protocol_id=protocol.protocol_id,
            dataset_digest=manifests[task].dataset_sha256,
            source_task=task,
            source_dataset_manifest_digest=manifest_digest,
            computation_backend=config.reducer.optimizer_backend,
        )
        reduced = reduce_kme(empirical, reducer_config)
        if reduced.reduction_error > config.reducer.reconstruction_tolerance:
            raise CommandFailure(
                f"{task} RKME error {reduced.reduction_error} exceeds tolerance "
                f"{config.reducer.reconstruction_tolerance}"
            )
        rkme_digest = reduced.save_npz(layout.task_rkme(task), overwrite=False)
        summary = {
            "schema": "policy-learnware.empirical-kme-summary.v0",
            "task": task,
            "protocol_id": protocol.protocol_id,
            "dataset_digest": manifests[task].dataset_sha256,
            "dataset_manifest_sha256": manifest_digest,
            "episode_count": empirical.episode_count,
            "transition_count": empirical.transition_count,
            "empirical_norm2": empirical.norm2,
            "kernel_bandwidth": empirical.bandwidth,
        }
        summary_digest = layout.publish_json(layout.empirical_summary(task), summary)
        task_manifest = {
            "schema": "policy-learnware.task-rkme.v0",
            "complete": True,
            "protocol_draft_hash": config.draft_hash,
            "protocol_id": protocol.protocol_id,
            "source_task": task,
            "source_dataset_digest": manifests[task].dataset_sha256,
            "source_dataset_manifest_digest": manifest_digest,
            "rkme_sha256": rkme_digest,
            "empirical_summary_sha256": summary_digest,
            "support_budget": reduced.supports.shape[0],
            "latent_dim": reduced.supports.shape[1],
            "reduction_error": reduced.reduction_error,
            "raw_reduction_residual_squared": reduced.raw_reduction_residual_squared,
            "negative_residual_clamped": reduced.negative_residual_clamped,
        }
        task_manifest_digest = layout.publish_json(
            layout.task_rkme_manifest(task), task_manifest
        )
        published[task] = {
            "rkme_sha256": rkme_digest,
            "manifest_sha256": task_manifest_digest,
            "reduction_error": reduced.reduction_error,
        }
        empirical_sources[task] = empirical
        reduced_sources[task] = reduced

    heldout_semantic, heldout_manifests = _load_semantic_split(
        layout, config, "separability_calibration"
    )
    ranking_records: dict[str, Any] = {}
    top1_agreements = 0
    pairwise_agreements: list[float] = []
    for query_task in config.environment.tasks:
        query = build_empirical_kme(
            heldout_semantic[query_task],
            kernel,
            protocol_id=protocol.protocol_id,
            dataset_digest=heldout_manifests[query_task].dataset_sha256,
            computation_backend=config.reducer.optimizer_backend,
        )
        unreduced_distances = {
            candidate: float(
                np.sqrt(
                    empirical_mmd2(
                        query,
                        empirical_sources[candidate],
                        computation_backend=config.reducer.optimizer_backend,
                    )
                )
            )
            for candidate in config.environment.tasks
        }
        reduced_distances = {
            candidate: empirical_to_reduced_distance(
                query,
                reduced_sources[candidate],
                negative_tolerance=config.selector.negative_tolerance,
            ).distance
            for candidate in config.environment.tasks
        }
        unreduced_ranking = deterministic_ranking(unreduced_distances)
        reduced_ranking = deterministic_ranking(reduced_distances)
        top1_agrees = unreduced_ranking[0] == reduced_ranking[0]
        pairwise_agreement = pairwise_order_agreement(
            unreduced_ranking, reduced_ranking
        )
        top1_agreements += int(top1_agrees)
        pairwise_agreements.append(pairwise_agreement)
        ranking_records[query_task] = {
            "query_dataset_sha256": heldout_manifests[query_task].dataset_sha256,
            "unreduced_distances": unreduced_distances,
            "reduced_distances": reduced_distances,
            "unreduced_ranking": list(unreduced_ranking),
            "reduced_ranking": list(reduced_ranking),
            "unreduced_top1": unreduced_ranking[0],
            "reduced_top1": reduced_ranking[0],
            "top1_agrees": top1_agrees,
            "pairwise_order_agreement": pairwise_agreement,
        }
    top1_agreement = top1_agreements / len(config.environment.tasks)
    ranking_decision = ranking_gate(
        top1_agreement=top1_agreement,
        minimum_top1_agreement=(
            config.gates.reduced_unreduced_ranking.minimum_top1_agreement
        ),
    )
    ranking_payload = {
        "schema": "policy-learnware.reduced-unreduced-ranking.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "source_split": "source_taskspec",
        "query_split": "separability_calibration",
        "source_query_splits_are_distinct": True,
        "source_dataset_manifest_sha256": {
            task: sha256_file(layout.dataset_manifest("source_taskspec", task))
            for task in config.environment.tasks
        },
        "query_dataset_manifest_sha256": {
            task: sha256_file(
                layout.dataset_manifest("separability_calibration", task)
            )
            for task in config.environment.tasks
        },
        "task_rkme_sha256": {
            task: sha256_file(layout.task_rkme(task))
            for task in config.environment.tasks
        },
        "top1_agreement": top1_agreement,
        "mean_pairwise_order_agreement": float(np.mean(pairwise_agreements)),
        "queries": ranking_records,
        "gate": ranking_decision.to_dict(),
        "gate_passed": ranking_decision.passed,
    }
    ranking_digest = layout.publish_json(
        layout.reduced_unreduced_ranking, ranking_payload
    )
    _validate_ranking_artifact(
        ranking_payload,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    return {
        "task_count": len(published),
        "task_specs": published,
        "ranking_gate": str(layout.reduced_unreduced_ranking),
        "ranking_gate_sha256": ranking_digest,
        "ranking_gate_passed": ranking_decision.passed,
    }


def _validated_jobs_manifest(
    config: ProtocolDraft,
) -> tuple[Path, tuple[str, ...]]:
    path = _jobs_manifest_path(config)
    payload = read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("full"), list):
        raise CommandFailure("reproduction jobs_manifest.json has no full job list")
    expected_matrix = {
        (task, algorithm, seed)
        for task in config.environment.tasks
        for algorithm in ("ppo", "fpo")
        for seed in range(5)
    }
    actual_matrix: set[tuple[str, str, int]] = set()
    job_ids: list[str] = []
    for raw in payload["full"]:
        if not isinstance(raw, Mapping):
            raise CommandFailure("full jobs manifest contains a non-object")
        try:
            job_id = str(raw["job_id"])
            task = str(raw["task"])
            algorithm = str(raw["algorithm"])
            seed = int(raw["seed"])
            export_outers = {int(value) for value in raw["export_outers"]}
        except (KeyError, TypeError, ValueError) as error:
            raise CommandFailure(f"invalid full job record: {raw}") from error
        if config.pool.checkpoint_outer not in export_outers:
            raise CommandFailure(f"job {job_id} does not export the fixed pool outer")
        if "phase" in raw and raw["phase"] != "full":
            raise CommandFailure(f"job {job_id} declares a non-full phase")
        expected_job_id = f"full__{task}__{algorithm}__seed{seed}"
        if job_id != expected_job_id:
            raise CommandFailure(
                f"job id {job_id!r} differs from canonical {expected_job_id!r}"
            )
        actual_matrix.add((task, algorithm, seed))
        job_ids.append(job_id)
    if len(job_ids) != len(set(job_ids)):
        raise CommandFailure("reproduction jobs manifest has duplicate job ids")
    if actual_matrix != expected_matrix or len(job_ids) != len(expected_matrix):
        raise CommandFailure("reproduction full-job matrix is not six tasks × PPO/FPO × seeds 0..4")
    return path, tuple(job_ids)


def _inventory_payload(
    report: InventoryReport,
    *,
    config: ProtocolDraft,
    jobs_manifest: Path,
    runs_root: Path,
) -> dict[str, Any]:
    return {
        "schema": INVENTORY_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "checkpoint_outer": report.checkpoint_outer,
        "actual_environment_steps": config.pool.actual_environment_steps,
        "runs_root": str(runs_root),
        "jobs_manifest": str(jobs_manifest),
        "jobs_manifest_sha256": sha256_file(jobs_manifest),
        "items": [
            {
                "job_id": item.job_id,
                "attempt": item.attempt,
                "bundle_dir": str(item.metadata.bundle_dir),
                "bundle_digest": item.metadata.bundle_digest,
                "task": item.metadata.task,
                "algorithm": item.metadata.algorithm,
                "training_seed": item.metadata.training_seed,
                "outer_iteration": item.metadata.outer_iteration,
                "environment_steps": item.metadata.environment_steps,
                "observation_dim": item.metadata.observation_dim,
                "action_dim": item.metadata.action_dim,
            }
            for item in report.items
        ],
        "rejected": [asdict(item) for item in report.rejected],
    }


def _handle_inventory_policies(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outer = config.pool.checkpoint_outer if args.outer is None else args.outer
    if outer != config.pool.checkpoint_outer:
        raise CommandFailure(
            f"inventory outer {outer} differs from fixed config outer {config.pool.checkpoint_outer}"
        )
    if args.resume and layout.policy_inventory.is_file():
        payload = _load_inventory(layout, config)
        jobs_manifest = _jobs_manifest_path(config)
        if (
            Path(str(payload.get("jobs_manifest", ""))).resolve()
            != jobs_manifest.resolve()
            or payload.get("jobs_manifest_sha256") != sha256_file(jobs_manifest)
        ):
            raise ArtifactLayoutError("inventory resume jobs manifest binding mismatch")
        return {
            "resumed": True,
            "inventory": str(layout.policy_inventory),
            "item_count": len(payload.get("items", [])),
        }
    _assert_output_state([layout.policy_inventory], resume=args.resume)
    jobs_manifest, expected_job_ids = _validated_jobs_manifest(config)
    runs_root = (
        args.runs_root.expanduser().resolve()
        if args.runs_root is not None
        else _default_runs_root(config)
    )
    if not (runs_root / "full").is_dir():
        raise CommandFailure(f"full reproduction runs directory is missing: {runs_root / 'full'}")
    report = scan_policy_inventory(
        runs_root,
        checkpoint_outer=outer,
        expected_environment_steps=config.pool.actual_environment_steps,
        expected_job_ids=expected_job_ids,
    )
    if report.rejected:
        reasons = "; ".join(f"{item.job_id}: {item.reason}" for item in report.rejected)
        raise CommandFailure(f"policy inventory rejected candidates: {reasons}")
    expected_total = len(config.environment.tasks) * config.pool.candidates_per_task
    if len(report.items) != expected_total:
        raise CommandFailure(f"inventory has {len(report.items)} items, expected {expected_total}")
    counts = Counter(item.metadata.task for item in report.items)
    if any(counts[task] != config.pool.candidates_per_task for task in config.environment.tasks):
        raise CommandFailure(f"per-task candidate coverage mismatch: {dict(counts)}")
    payload = _inventory_payload(
        report, config=config, jobs_manifest=jobs_manifest, runs_root=runs_root
    )
    digest = layout.publish_json(layout.policy_inventory, payload)
    return {
        "inventory": str(layout.policy_inventory),
        "sha256": digest,
        "item_count": len(report.items),
        "per_task": dict(sorted(counts.items())),
    }


def _load_inventory(layout: ArtifactLayout, config: ProtocolDraft) -> Mapping[str, Any]:
    payload = read_json(layout.policy_inventory)
    if not isinstance(payload, Mapping) or payload.get("schema") != INVENTORY_SCHEMA:
        raise CommandFailure("unsupported policy inventory artifact")
    if payload.get("complete") is not True or payload.get("rejected"):
        raise CommandFailure("policy inventory is not complete and clean")
    if payload.get("protocol_draft_hash") != config.draft_hash:
        raise CommandFailure("policy inventory config hash mismatch")
    if int(payload.get("checkpoint_outer", -1)) != config.pool.checkpoint_outer:
        raise CommandFailure("policy inventory checkpoint outer mismatch")
    items = payload.get("items")
    if not isinstance(items, list):
        raise CommandFailure("policy inventory items must be a list")
    expected = len(config.environment.tasks) * config.pool.candidates_per_task
    if len(items) != expected:
        raise CommandFailure(f"policy inventory item count {len(items)} != {expected}")
    expected_matrix = {
        (task, algorithm, seed)
        for task in config.environment.tasks
        for algorithm in ("ppo", "fpo")
        for seed in range(5)
    }
    actual_matrix: set[tuple[str, str, int]] = set()
    job_ids: set[str] = set()
    for raw in items:
        if not isinstance(raw, Mapping):
            raise CommandFailure("policy inventory contains a non-object item")
        try:
            key = (
                str(raw["task"]),
                str(raw["algorithm"]),
                int(raw["training_seed"]),
            )
            job_id = str(raw["job_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise CommandFailure("policy inventory item has an invalid identity") from error
        if key in actual_matrix:
            raise CommandFailure(f"policy inventory has a duplicate candidate identity: {key}")
        if job_id in job_ids:
            raise CommandFailure(f"policy inventory has a duplicate job id: {job_id}")
        actual_matrix.add(key)
        job_ids.add(job_id)
    if actual_matrix != expected_matrix:
        raise CommandFailure("policy inventory candidate matrix is not tasks x PPO/FPO x seeds 0..4")
    return payload


def _handle_verify_policy_bundles(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    atol = config.policy.parity_atol if args.atol is None else args.atol
    rtol = config.policy.parity_rtol if args.rtol is None else args.rtol
    inventory = _load_inventory(layout, config)
    if args.resume and layout.bundle_verification.is_file():
        resumed = _resume_manifest(
            layout, layout.bundle_verification, config_hash=config.draft_hash
        )
        if resumed is not None:
            if resumed.get("inventory_sha256") != sha256_file(layout.policy_inventory):
                raise ArtifactLayoutError("verification resume inventory digest mismatch")
            if resumed.get("environment_manifest_sha256") != sha256_file(
                layout.environment_manifest
            ):
                raise ArtifactLayoutError(
                    "verification resume environment digest mismatch"
                )
            return {
                "resumed": True,
                "verification": str(layout.bundle_verification),
                "verified_count": resumed.get("verified_count"),
            }
    items = inventory["items"]
    report_paths = [layout.parity_report(str(item["job_id"])) for item in items]
    if not args.resume:
        _assert_output_state([*report_paths, layout.bundle_verification], resume=False)
    runtime_source = _verify_fpo_checkout(config)
    _configure_jax_runtime()
    schemas = _load_env_schemas(layout, config)
    runs_root = Path(str(inventory["runs_root"])).resolve()
    reports: list[tuple[Path, dict[str, Any]]] = []
    failures: list[str] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise CommandFailure("policy inventory contains a non-object item")
        job_id = str(raw["job_id"])
        bundle_dir = Path(str(raw["bundle_dir"])).resolve()
        try:
            bundle_dir.relative_to(runs_root)
        except ValueError as error:
            raise CommandFailure(f"inventory bundle escapes reproduction runs root: {bundle_dir}") from error
        try:
            metadata = validate_bundle(
                bundle_dir,
                expected_task=str(raw["task"]),
                expected_algorithm=str(raw["algorithm"]),
                expected_seed=int(raw["training_seed"]),
                expected_outer=config.pool.checkpoint_outer,
                expected_environment_steps=config.pool.actual_environment_steps,
            )
            if metadata.bundle_digest != str(raw["bundle_digest"]):
                raise BundleValidationError("bundle digest changed after inventory")
            schema = schemas[metadata.task]
            if (
                metadata.observation_dim != schema.observation_dim
                or metadata.action_dim != schema.action_dim
            ):
                raise BundleValidationError("policy bundle native dimensions disagree with EnvSchema")
            policy = load_policy(metadata, fpo_root=config.runtime.fpo_root)
            parity = verify_golden_parity(
                policy, metadata, atol=atol, rtol=rtol
            )
            if not parity.passed:
                raise BundleValidationError(
                    "golden parity failed: "
                    f"raw_error={parity.raw_max_abs_error}, "
                    f"environment_error={parity.environment_max_abs_error}"
                )
            report = {
                "schema": "policy-learnware.policy-parity-report.v0",
                "protocol_draft_hash": config.draft_hash,
                "job_id": job_id,
                "bundle_digest": metadata.bundle_digest,
                "bundle_dir": str(metadata.bundle_dir),
                **asdict(parity),
            }
            reports.append((layout.parity_report(job_id), report))
        except Exception as error:
            failures.append(f"{job_id}: {type(error).__name__}: {error}")
    if failures:
        raise CommandFailure("bundle verification failed closed: " + "; ".join(failures))

    file_records: dict[str, dict[str, str]] = {}
    for report_path, report in reports:
        digest = layout.publish_json(report_path, report, resume=args.resume)
        file_records[report_path.stem] = _manifest_file(layout, report_path, digest)
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "inventory_sha256": sha256_file(layout.policy_inventory),
        "environment_manifest_sha256": sha256_file(layout.environment_manifest),
        "runtime_source": runtime_source,
        "atol": atol,
        "rtol": rtol,
        "verified_count": len(reports),
        "files": file_records,
    }
    digest = layout.publish_json(
        layout.bundle_verification, verification, resume=args.resume
    )
    return {
        "verification": str(layout.bundle_verification),
        "sha256": digest,
        "verified_count": len(reports),
    }


CHAMPIONIZATION_CANDIDATE_SCHEMA = (
    "policy-learnware.championization-candidate.v0"
)
CHAMPIONIZATION_WORKER_ENV = "POLICY_LEARNWARE_CHAMPIONIZATION_WORKER"


@contextmanager
def _exclusive_championization_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CommandFailure(
                f"another championization parent owns the pool lock: {path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _championization_seed_vectors(
    item: Mapping[str, Any], config: ProtocolDraft
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    task = str(item["task"])
    task_index = config.environment.tasks.index(task)
    count = config.episodes.championization_per_candidate
    plan = SeedPlan(config.project_seed)
    return (
        tuple(
            plan.derive(
                "championization",
                task_index,
                index,
                stream="environment_reset",
            )
            for index in range(count)
        ),
        tuple(
            plan.derive(
                "championization",
                task_index,
                index,
                stream="policy_action",
            )
            for index in range(count)
        ),
    )


def _championization_evaluator_contract() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    sources = {
        relative: sha256_file(package_root / relative)
        for relative in (
            "policy/evaluate.py",
            "policy/loader.py",
            "policy/parity.py",
            "envs/mujoco_playground.py",
        )
    }
    return {
        "schema": "policy-learnware.championization-evaluator.v0",
        "source_sha256": sources,
        "contract_sha256": sha256_json(
            {
                "sources": sources,
                "execution": "jax-jit-lax-map-lax-scan",
                "policy_deterministic_argument": True,
                "action_transform": "tanh(raw_action)",
                "return_accumulation": "host-float64-left-to-right",
            }
        ),
        "runtime": {
            "python": sys.version,
            **_runtime_package_versions(),
        },
        "execution": "jax-jit-lax-map-lax-scan",
        "policy_deterministic_argument": True,
        "action_transform": "tanh(raw_action)",
        "return_accumulation": "host-float64-left-to-right",
    }


def _validate_championization_candidate_shard(
    payload: Any,
    *,
    item: Mapping[str, Any],
    candidate_index: int,
    config: ProtocolDraft,
    inventory_sha256: str,
    verification_sha256: str,
    environment_manifest_sha256: str,
    evaluator_contract: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = str(item["job_id"])
    reset_seeds, policy_seeds = _championization_seed_vectors(item, config)
    if not isinstance(payload, Mapping):
        raise ArtifactLayoutError(
            f"championization candidate shard is not an object: {job_id}"
        )
    if (
        payload.get("schema") != CHAMPIONIZATION_CANDIDATE_SCHEMA
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("inventory_sha256") != inventory_sha256
        or payload.get("verification_sha256") != verification_sha256
        or payload.get("environment_manifest_sha256")
        != environment_manifest_sha256
        or int(payload.get("candidate_index", -1)) != candidate_index
        or str(payload.get("job_id", "")) != job_id
        or str(payload.get("bundle_digest", ""))
        != str(item["bundle_digest"])
        or str(payload.get("task", "")) != str(item["task"])
        or str(payload.get("algorithm", "")) != str(item["algorithm"])
        or int(payload.get("training_seed", -1)) != int(item["training_seed"])
        or list(payload.get("reset_seeds", [])) != list(reset_seeds)
        or list(payload.get("policy_seeds", [])) != list(policy_seeds)
        or canonicalize(payload.get("evaluator_contract"))
        != canonicalize(evaluator_contract)
    ):
        raise ArtifactLayoutError(
            f"championization candidate shard binding mismatch: {job_id}"
        )
    episode_returns = payload.get("episode_returns")
    parity = payload.get("parity")
    compiled_parity = payload.get("compiled_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or float(parity.get("atol", -1.0)) != config.policy.parity_atol
        or float(parity.get("rtol", -1.0)) != config.policy.parity_rtol
    ):
        raise ArtifactLayoutError(
            f"championization candidate parity record mismatch: {job_id}"
        )
    if (
        not isinstance(compiled_parity, Mapping)
        or compiled_parity.get("passed") is not True
        or compiled_parity.get("next_keys_equal") is not True
        or float(compiled_parity.get("atol", -1.0)) != config.policy.parity_atol
        or float(compiled_parity.get("rtol", -1.0)) != config.policy.parity_rtol
    ):
        raise ArtifactLayoutError(
            f"championization compiled parity record mismatch: {job_id}"
        )
    count = config.episodes.championization_per_candidate
    if not isinstance(episode_returns, list) or len(episode_returns) != count:
        raise ArtifactLayoutError(
            f"championization candidate shard return count mismatch: {job_id}"
        )
    try:
        parsed_returns = [float(value) for value in episode_returns]
    except (TypeError, ValueError) as error:
        raise ArtifactLayoutError(
            f"championization candidate shard has invalid returns: {job_id}"
        ) from error
    if any(not np.isfinite(value) for value in parsed_returns):
        raise ArtifactLayoutError(
            f"championization candidate shard has non-finite returns: {job_id}"
        )
    return {
        "job_id": job_id,
        "bundle_digest": str(item["bundle_digest"]),
        "task": str(item["task"]),
        "algorithm": str(item["algorithm"]),
        "training_seed": int(item["training_seed"]),
        "reset_seeds": list(reset_seeds),
        "policy_seeds": list(policy_seeds),
        "episode_returns": parsed_returns,
        "evaluator_contract": canonicalize(evaluator_contract),
        "parity": canonicalize(parity),
        "compiled_parity": canonicalize(compiled_parity),
    }


def _evaluate_championization_candidate(
    *,
    item: Mapping[str, Any],
    candidate_index: int,
    shard_index: int,
    shard_count: int,
    config: ProtocolDraft,
    layout: ArtifactLayout,
    schemas: Mapping[str, EnvSchema],
    inventory_sha256: str,
    verification_sha256: str,
    environment_manifest_sha256: str,
    evaluator_contract: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    job_id = str(item["job_id"])
    destination = layout.championization_candidate(job_id)
    if destination.is_file():
        if not resume:
            raise ArtifactExistsError(
                f"refusing to overwrite artifact: {destination}"
            )
        return (
            _validate_championization_candidate_shard(
                read_json(destination),
                item=item,
                candidate_index=candidate_index,
                config=config,
                inventory_sha256=inventory_sha256,
                verification_sha256=verification_sha256,
                environment_manifest_sha256=environment_manifest_sha256,
                evaluator_contract=evaluator_contract,
            ),
            True,
        )

    task = str(item["task"])
    started = time.monotonic()
    print(
        f"[championize shard {shard_index + 1}/{shard_count}] "
        f"start {candidate_index + 1}: {job_id}",
        file=sys.stderr,
        flush=True,
    )
    metadata = validate_bundle(
        Path(str(item["bundle_dir"])),
        expected_task=task,
        expected_algorithm=str(item["algorithm"]),
        expected_seed=int(item["training_seed"]),
        expected_outer=config.pool.checkpoint_outer,
        expected_environment_steps=config.pool.actual_environment_steps,
    )
    policy = load_policy(metadata, fpo_root=config.runtime.fpo_root)
    parity = verify_golden_parity(
        policy,
        metadata,
        atol=config.policy.parity_atol,
        rtol=config.policy.parity_rtol,
    )
    if not parity.passed:
        raise CommandFailure(
            "golden parity failed immediately before championization: "
            f"{job_id}; raw_error={parity.raw_max_abs_error}, "
            f"environment_error={parity.environment_max_abs_error}, "
            f"atol={parity.atol}, rtol={parity.rtol}"
        )
    with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as golden:
        compiled_parity = verify_compiled_policy_parity(
            policy,
            np.asarray(golden["observation"]),
            np.asarray(golden["prng_key_data"]),
            atol=config.policy.parity_atol,
            rtol=config.policy.parity_rtol,
        )
    if not compiled_parity.passed:
        raise CommandFailure(
            "compiled evaluator parity failed immediately before "
            f"championization: {job_id}; max_error="
            f"{compiled_parity.max_abs_error}"
        )
    reset_seeds, policy_seeds = _championization_seed_vectors(item, config)
    episode_returns = _evaluate_frozen_policy_returns_accelerated(
        policy,
        task=task,
        reset_seeds=reset_seeds,
        policy_seeds=policy_seeds,
        config=config,
        expected_schema=schemas[task],
    )
    shard_payload = {
        "schema": CHAMPIONIZATION_CANDIDATE_SCHEMA,
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "inventory_sha256": inventory_sha256,
        "verification_sha256": verification_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "candidate_index": candidate_index,
        "job_id": job_id,
        "bundle_digest": metadata.bundle_digest,
        "task": task,
        "algorithm": metadata.algorithm,
        "training_seed": metadata.training_seed,
        "reset_seeds": list(reset_seeds),
        "policy_seeds": list(policy_seeds),
        "episode_returns": list(episode_returns),
        "parity": asdict(parity),
        "compiled_parity": asdict(compiled_parity),
        "evaluator_contract": dict(evaluator_contract),
        "episode_batch_size": len(reset_seeds),
        "fixed_horizon": schemas[task].horizon,
    }
    layout.publish_json(destination, shard_payload)
    validated = _validate_championization_candidate_shard(
        shard_payload,
        item=item,
        candidate_index=candidate_index,
        config=config,
        inventory_sha256=inventory_sha256,
        verification_sha256=verification_sha256,
        environment_manifest_sha256=environment_manifest_sha256,
        evaluator_contract=evaluator_contract,
    )
    print(
        f"[championize shard {shard_index + 1}/{shard_count}] "
        f"complete {job_id} in {time.monotonic() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return validated, False


def _run_championization_shard(
    *,
    inventory: Mapping[str, Any],
    shard_index: int,
    shard_count: int,
    config: ProtocolDraft,
    layout: ArtifactLayout,
    resume: bool,
) -> dict[str, Any]:
    _configure_jax_runtime()
    schemas = _load_env_schemas(layout, config)
    inventory_sha256 = sha256_file(layout.policy_inventory)
    verification_sha256 = sha256_file(layout.bundle_verification)
    environment_manifest_sha256 = sha256_file(layout.environment_manifest)
    evaluator_contract = _championization_evaluator_contract()
    selected = [
        (index, item)
        for index, item in enumerate(inventory["items"])
        if index % shard_count == shard_index
    ]
    resumed_count = 0
    for candidate_index, item in selected:
        _, resumed = _evaluate_championization_candidate(
            item=item,
            candidate_index=candidate_index,
            shard_index=shard_index,
            shard_count=shard_count,
            config=config,
            layout=layout,
            schemas=schemas,
            inventory_sha256=inventory_sha256,
            verification_sha256=verification_sha256,
            environment_manifest_sha256=environment_manifest_sha256,
            evaluator_contract=evaluator_contract,
            resume=resume,
        )
        resumed_count += int(resumed)
        if not resumed:
            # Each candidate closes over different frozen weights.  Release its
            # executable before compiling the next candidate so long-lived GPU
            # workers do not retain all 7-8 candidate programs at once.
            try:
                import jax

                jax.clear_caches()
            except (ImportError, AttributeError):  # pragma: no cover
                pass
            gc.collect()
    return {
        "worker": True,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "candidate_count": len(selected),
        "resumed_candidate_count": resumed_count,
    }


def _resolve_championization_devices(raw: str) -> tuple[str, ...]:
    requested = raw.strip()
    if requested == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            devices = tuple(part.strip() for part in visible.split(",") if part.strip())
            if not devices or devices == ("-1",):
                raise CommandFailure(
                    "CUDA_VISIBLE_DEVICES exposes no devices for championization"
                )
        else:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise CommandFailure(f"cannot discover CUDA devices: {detail}")
            devices = tuple(
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            )
    else:
        devices = tuple(part.strip() for part in requested.split(","))
        if not devices or any(not device or not device.isdecimal() for device in devices):
            raise CommandFailure("--devices must be 'auto' or comma-separated integer ids")
    if any(not device or any(character.isspace() for character in device) for device in devices):
        raise CommandFailure("resolved CUDA devices contain an invalid token")
    if len(set(devices)) != len(devices):
        raise CommandFailure("--devices contains duplicates")
    return devices


def _parse_championization_worker_output(stdout: str) -> Mapping[str, Any]:
    """Extract one structured CLI record from third-party stdout noise.

    MuJoCo may print optional-backend import warnings directly to stdout before
    our CLI writes its JSON result.  The worker protocol therefore scans JSON
    values and accepts exactly one top-level championization completion record;
    it never treats arbitrary text or a second record as success.
    """

    decoder = json.JSONDecoder()
    records: list[Mapping[str, Any]] = []
    offset = 0
    while True:
        start = stdout.find("{", offset)
        if start < 0:
            break
        try:
            candidate, consumed = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = start + max(consumed, 1)
        if (
            isinstance(candidate, Mapping)
            and candidate.get("schema") == CLI_SCHEMA
            and candidate.get("command") == "championize"
        ):
            records.append(candidate)
    if len(records) != 1:
        raise CommandFailure(
            "championization shard emitted no unique CLI completion record"
        )
    return records[0]


def _launch_championization_workers(
    *,
    devices: Sequence[str],
    args: argparse.Namespace,
    config: ProtocolDraft,
) -> None:
    processes: dict[int, tuple[str, subprocess.Popen[str]]] = {}
    base_command = [
        sys.executable,
        "-m",
        "policy_learnware_v0.cli",
        "championize",
        "--config",
        str(args.config.resolve()),
        "--artifacts-root",
        str(args.artifacts_root.expanduser().resolve()),
        "--outer",
        str(config.pool.checkpoint_outer),
    ]
    if args.resume:
        base_command.append("--resume")
    pending: set[int] = set()
    try:
        for shard_index, device in enumerate(devices):
            command = [
                *base_command,
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(devices)),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment[CHAMPIONIZATION_WORKER_ENV] = "1"
            environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            environment.setdefault("WANDB_MODE", "disabled")
            processes[shard_index] = (
                device,
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=None,
                    text=True,
                ),
            )
            pending.add(shard_index)
            print(
                f"[championize parent] launched shard {shard_index + 1}/"
                f"{len(devices)} on CUDA device {device}",
                file=sys.stderr,
                flush=True,
            )
        while pending:
            for shard_index in tuple(sorted(pending)):
                device, process = processes[shard_index]
                returncode = process.poll()
                if returncode is None:
                    continue
                stdout, _ = process.communicate()
                pending.remove(shard_index)
                if returncode != 0:
                    detail = stdout.strip()[-4000:]
                    raise CommandFailure(
                        f"championization shard {shard_index} on CUDA device "
                        f"{device} failed with exit {returncode}: {detail}"
                    )
                child_payload = _parse_championization_worker_output(stdout)
                child_result = child_payload.get("result", {})
                if (
                    child_payload.get("status") != "ok"
                    or not isinstance(child_result, Mapping)
                    or child_result.get("worker") is not True
                    or int(child_result.get("shard_index", -1)) != shard_index
                    or int(child_result.get("shard_count", -1)) != len(devices)
                ):
                    raise CommandFailure(
                        f"championization shard {shard_index} completion record "
                        "is malformed"
                    )
                print(
                    f"[championize parent] shard {shard_index + 1}/"
                    f"{len(devices)} completed on CUDA device {device}",
                    file=sys.stderr,
                    flush=True,
                )
            if pending:
                time.sleep(0.2)
    except BaseException:
        for shard_index in pending:
            process = processes[shard_index][1]
            if process.poll() is None:
                process.terminate()
        for shard_index in pending:
            process = processes[shard_index][1]
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        raise


def _merge_championization_candidate_shards(
    *,
    inventory: Mapping[str, Any],
    config: ProtocolDraft,
    layout: ArtifactLayout,
) -> list[dict[str, Any]]:
    inventory_sha256 = sha256_file(layout.policy_inventory)
    verification_sha256 = sha256_file(layout.bundle_verification)
    environment_manifest_sha256 = sha256_file(layout.environment_manifest)
    evaluator_contract = _championization_evaluator_contract()
    merged: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate_index, item in enumerate(inventory["items"]):
        path = layout.championization_candidate(str(item["job_id"]))
        if not path.is_file():
            missing.append(str(item["job_id"]))
            continue
        merged.append(
            _validate_championization_candidate_shard(
                read_json(path),
                item=item,
                candidate_index=candidate_index,
                config=config,
                inventory_sha256=inventory_sha256,
                verification_sha256=verification_sha256,
                environment_manifest_sha256=environment_manifest_sha256,
                evaluator_contract=evaluator_contract,
            )
        )
    if missing:
        raise CommandFailure(
            "championization candidate shards are incomplete: "
            + ", ".join(missing)
        )
    return merged


def _championization_summary_payload(
    result: ChampionizationResult,
    *,
    config: ProtocolDraft,
    layout: ArtifactLayout,
) -> dict[str, Any]:
    return {
        "schema": "policy-learnware.championization.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "pool_id": config.pool.pool_id,
        "returns_sha256": sha256_file(layout.championization_returns),
        "inventory_sha256": sha256_file(layout.policy_inventory),
        "verification_sha256": sha256_file(layout.bundle_verification),
        "environment_manifest_sha256": sha256_file(layout.environment_manifest),
        "episode_count_per_candidate": config.episodes.championization_per_candidate,
        "evaluator_contract": _championization_evaluator_contract(),
        "selection_rule": result.selection_rule,
        "tasks": {
            champion.task: {
                "selected_bundle_dir": str(champion.selected.candidate.bundle_dir),
                "selected_bundle_digest": champion.selected.candidate.bundle_digest,
                "selected_algorithm": champion.selected.candidate.algorithm,
                "selected_training_seed": champion.selected.candidate.training_seed,
                "selected_mean_return": champion.selected.mean_return,
                "selected_return_std": champion.selected.return_std,
                "ranking": [
                    {
                        "bundle_dir": str(item.candidate.bundle_dir),
                        "bundle_digest": item.candidate.bundle_digest,
                        "algorithm": item.candidate.algorithm,
                        "training_seed": item.candidate.training_seed,
                        "mean_return": item.mean_return,
                        "return_std": item.return_std,
                        "episode_returns": list(item.candidate.episode_returns),
                    }
                    for item in champion.ranking
                ],
            }
            for champion in result.champions
        },
    }


def _handle_championize_unlocked(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    missing_inputs = [
        path
        for path in (
            layout.policy_inventory,
            layout.bundle_verification,
        )
        if not path.is_file()
    ]
    if missing_inputs:
        raise CommandFailure(
            "championization prerequisites are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )
    inventory = _load_inventory(layout, config)
    verification = _resume_manifest(
        layout, layout.bundle_verification, config_hash=config.draft_hash
    )
    if verification is None:
        raise CommandFailure("bundle verification completion manifest is missing")
    if verification.get("inventory_sha256") != sha256_file(layout.policy_inventory):
        raise CommandFailure("bundle verification inventory digest mismatch")
    if verification.get("environment_manifest_sha256") != sha256_file(
        layout.environment_manifest
    ):
        raise CommandFailure("bundle verification environment digest mismatch")
    if args.resume and layout.championization.is_file():
        payload = read_json(layout.championization)
        if (
            isinstance(payload, Mapping)
            and payload.get("complete") is True
            and payload.get("protocol_draft_hash") == config.draft_hash
            and layout.championization_returns.is_file()
            and payload.get("returns_sha256")
            == sha256_file(layout.championization_returns)
            and payload.get("inventory_sha256")
            == sha256_file(layout.policy_inventory)
            and payload.get("verification_sha256")
            == sha256_file(layout.bundle_verification)
            and payload.get("environment_manifest_sha256")
            == sha256_file(layout.environment_manifest)
        ):
            recomputed = _recompute_championization_from_artifacts(layout, config)
            expected = _championization_summary_payload(
                recomputed, config=config, layout=layout
            )
            if canonicalize(payload) != canonicalize(expected):
                raise ArtifactLayoutError(
                    "championization resume summary differs from recomputed returns"
                )
            return {"resumed": True, "championization": str(layout.championization)}
        raise ArtifactLayoutError("championization resume validation failed")
    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    candidate_paths = [
        layout.championization_candidate(str(item["job_id"]))
        for index, item in enumerate(inventory["items"])
        if shard_index is None or index % shard_count == shard_index
    ]
    if not args.resume:
        aggregate_outputs = (
            [layout.championization_returns, layout.championization]
            if shard_index is None
            else []
        )
        _assert_output_state(
            [*candidate_paths, *aggregate_outputs],
            resume=False,
        )
    if shard_index is not None and shard_count is not None:
        return _run_championization_shard(
            inventory=inventory,
            shard_index=shard_index,
            shard_count=shard_count,
            config=config,
            layout=layout,
            resume=args.resume,
        )
    if not layout.championization_returns.is_file():
        devices_raw = getattr(args, "devices", None)
        if getattr(args, "merge_only", False):
            pass
        elif devices_raw is not None:
            devices = _resolve_championization_devices(devices_raw)
            _launch_championization_workers(
                devices=devices,
                args=args,
                config=config,
            )
        else:
            _run_championization_shard(
                inventory=inventory,
                shard_index=0,
                shard_count=1,
                config=config,
                layout=layout,
                resume=args.resume,
            )
        generated_candidates = _merge_championization_candidate_shards(
            inventory=inventory,
            config=config,
            layout=layout,
        )
        layout.publish_json(
            layout.championization_returns,
            {
                "schema": "policy-learnware.championization-returns.v0",
                "complete": True,
                "protocol_draft_hash": config.draft_hash,
                "inventory_sha256": sha256_file(layout.policy_inventory),
                "verification_sha256": sha256_file(layout.bundle_verification),
                "environment_manifest_sha256": sha256_file(
                    layout.environment_manifest
                ),
                "episode_count_per_candidate": config.episodes.championization_per_candidate,
                "evaluator_contract": _championization_evaluator_contract(),
                "candidates": generated_candidates,
            },
        )
    returns_payload = read_json(layout.championization_returns)
    if (
        not isinstance(returns_payload, Mapping)
        or returns_payload.get("schema")
        != "policy-learnware.championization-returns.v0"
        or returns_payload.get("complete") is not True
    ):
        raise CommandFailure("unsupported/incomplete championization returns artifact")
    if returns_payload.get("protocol_draft_hash") != config.draft_hash:
        raise CommandFailure("championization returns config hash mismatch")
    if returns_payload.get("inventory_sha256") != sha256_file(layout.policy_inventory):
        raise CommandFailure("championization returns inventory digest mismatch")
    if returns_payload.get("verification_sha256") != sha256_file(
        layout.bundle_verification
    ):
        raise CommandFailure("championization returns verification digest mismatch")
    if returns_payload.get("environment_manifest_sha256") != sha256_file(
        layout.environment_manifest
    ):
        raise CommandFailure("championization returns environment digest mismatch")
    if int(returns_payload.get("episode_count_per_candidate", -1)) != int(
        config.episodes.championization_per_candidate
    ):
        raise CommandFailure("championization return episode count mismatch")
    if canonicalize(returns_payload.get("evaluator_contract")) != canonicalize(
        _championization_evaluator_contract()
    ):
        raise CommandFailure("championization evaluator contract mismatch")
    raw_results = returns_payload.get("candidates")
    if not isinstance(raw_results, list):
        raise CommandFailure("championization returns candidates must be a list")
    inventory_by_job = {str(item["job_id"]): item for item in inventory["items"]}
    if len(raw_results) != len(inventory_by_job):
        raise CommandFailure("championization return candidate count mismatch")
    plan = SeedPlan(config.project_seed)
    candidates: list[CandidateEvaluation] = []
    seen_jobs: set[str] = set()
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise CommandFailure("championization candidate result is not an object")
        job_id = str(raw.get("job_id", ""))
        if job_id in seen_jobs or job_id not in inventory_by_job:
            raise CommandFailure(f"unknown or duplicate championization job: {job_id}")
        seen_jobs.add(job_id)
        item = inventory_by_job[job_id]
        task = str(item["task"])
        if (
            str(raw.get("task", "")) != task
            or str(raw.get("algorithm", "")) != str(item["algorithm"])
            or int(raw.get("training_seed", -1)) != int(item["training_seed"])
        ):
            raise CommandFailure(f"championization identity mismatch for {job_id}")
        task_index = config.environment.tasks.index(task)
        count = config.episodes.championization_per_candidate
        expected_reset = [
            plan.derive(
                "championization",
                task_index,
                index,
                stream="environment_reset",
            )
            for index in range(count)
        ]
        expected_policy = [
            plan.derive(
                "championization", task_index, index, stream="policy_action"
            )
            for index in range(count)
        ]
        if list(raw.get("reset_seeds", [])) != expected_reset:
            raise CommandFailure(f"championization reset seeds differ for {job_id}")
        if list(raw.get("policy_seeds", [])) != expected_policy:
            raise CommandFailure(f"championization policy seeds differ for {job_id}")
        episode_returns = raw.get("episode_returns")
        if not isinstance(episode_returns, list) or len(episode_returns) != count:
            raise CommandFailure(
                f"{job_id} must contain exactly {count} championization returns"
            )
        if str(raw.get("bundle_digest", "")) != str(item["bundle_digest"]):
            raise CommandFailure(f"championization bundle digest mismatch for {job_id}")
        metadata = validate_bundle(
            Path(str(item["bundle_dir"])),
            expected_task=task,
            expected_algorithm=str(item["algorithm"]),
            expected_seed=int(item["training_seed"]),
            expected_outer=config.pool.checkpoint_outer,
            expected_environment_steps=config.pool.actual_environment_steps,
        )
        candidates.append(
            CandidateEvaluation.from_bundle(
                metadata, episode_returns, parity_passed=True
            )
        )
    result = championize(
        candidates,
        checkpoint_outer=config.pool.checkpoint_outer,
        expected_environment_steps=config.pool.actual_environment_steps,
        expected_candidates_per_task=config.pool.candidates_per_task,
        expected_tasks=config.environment.tasks,
    )
    if result.rejected:
        raise CommandFailure("championization rejected a verified fixed-budget candidate")
    payload = _championization_summary_payload(result, config=config, layout=layout)
    digest = layout.publish_json(layout.championization, payload)
    return {
        "championization": str(layout.championization),
        "sha256": digest,
        "champion_count": len(result.champions),
    }


def _handle_championize(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    if getattr(args, "shard_index", None) is not None:
        return _handle_championize_unlocked(args, config, layout)
    if not layout.policy_inventory.is_file() or not layout.bundle_verification.is_file():
        # Preserve the CLI's fail-without-writes prerequisite contract.
        return _handle_championize_unlocked(args, config, layout)
    with _exclusive_championization_lock(layout.championization_lock):
        return _handle_championize_unlocked(args, config, layout)


def _recompute_championization_from_artifacts(
    layout: ArtifactLayout,
    config: ProtocolDraft,
) -> ChampionizationResult:
    """Recompute the fixed winner rule from hash-bound private artifacts."""

    inventory = _load_inventory(layout, config)
    verification = _resume_manifest(
        layout, layout.bundle_verification, config_hash=config.draft_hash
    )
    inventory_digest = sha256_file(layout.policy_inventory)
    verification_digest = sha256_file(layout.bundle_verification)
    if (
        verification is None
        or verification.get("inventory_sha256") != inventory_digest
        or verification.get("environment_manifest_sha256")
        != sha256_file(layout.environment_manifest)
    ):
        raise CommandFailure("bundle verification is not bound to the current inventory")
    returns_payload = read_json(layout.championization_returns)
    if (
        not isinstance(returns_payload, Mapping)
        or returns_payload.get("schema")
        != "policy-learnware.championization-returns.v0"
        or returns_payload.get("complete") is not True
        or returns_payload.get("protocol_draft_hash") != config.draft_hash
        or returns_payload.get("inventory_sha256") != inventory_digest
        or returns_payload.get("verification_sha256") != verification_digest
        or returns_payload.get("environment_manifest_sha256")
        != sha256_file(layout.environment_manifest)
        or int(returns_payload.get("episode_count_per_candidate", -1))
        != config.episodes.championization_per_candidate
        or canonicalize(returns_payload.get("evaluator_contract"))
        != canonicalize(_championization_evaluator_contract())
    ):
        raise CommandFailure(
            "championization returns are incomplete or not bound to current inputs"
        )
    raw_candidates = returns_payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise CommandFailure("championization returns candidates must be a list")
    inventory_by_job = {
        str(item["job_id"]): item for item in inventory["items"]
    }
    if len(raw_candidates) != len(inventory_by_job):
        raise CommandFailure("championization return candidate count mismatch")
    seed_plan = SeedPlan(config.project_seed)
    candidates: list[CandidateEvaluation] = []
    seen_jobs: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, Mapping):
            raise CommandFailure("championization candidate result is not an object")
        job_id = str(raw.get("job_id", ""))
        if job_id in seen_jobs or job_id not in inventory_by_job:
            raise CommandFailure(f"unknown or duplicate championization job: {job_id}")
        seen_jobs.add(job_id)
        item = inventory_by_job[job_id]
        task = str(item["task"])
        algorithm = str(item["algorithm"])
        training_seed = int(item["training_seed"])
        if (
            str(raw.get("task", "")) != task
            or str(raw.get("algorithm", "")) != algorithm
            or int(raw.get("training_seed", -1)) != training_seed
            or str(raw.get("bundle_digest", "")) != str(item["bundle_digest"])
        ):
            raise CommandFailure(f"championization identity mismatch for {job_id}")
        task_index = config.environment.tasks.index(task)
        count = config.episodes.championization_per_candidate
        expected_reset = [
            seed_plan.derive(
                "championization", task_index, index, stream="environment_reset"
            )
            for index in range(count)
        ]
        expected_policy = [
            seed_plan.derive(
                "championization", task_index, index, stream="policy_action"
            )
            for index in range(count)
        ]
        episode_returns = raw.get("episode_returns")
        if (
            list(raw.get("reset_seeds", [])) != expected_reset
            or list(raw.get("policy_seeds", [])) != expected_policy
            or not isinstance(episode_returns, list)
            or len(episode_returns) != count
        ):
            raise CommandFailure(f"championization seeds/returns mismatch for {job_id}")
        candidates.append(
            CandidateEvaluation(
                task=task,
                algorithm=algorithm,
                training_seed=training_seed,
                outer_iteration=int(item["outer_iteration"]),
                environment_steps=int(item["environment_steps"]),
                bundle_dir=Path(str(item["bundle_dir"])),
                bundle_digest=str(item["bundle_digest"]),
                episode_returns=tuple(float(value) for value in episode_returns),
                checksum_passed=True,
                parity_passed=True,
            )
        )
    result = championize(
        candidates,
        checkpoint_outer=config.pool.checkpoint_outer,
        expected_environment_steps=config.pool.actual_environment_steps,
        expected_candidates_per_task=config.pool.candidates_per_task,
        expected_tasks=config.environment.tasks,
    )
    if result.rejected:
        raise CommandFailure("hash-bound championization candidates were rejected")
    return result


def _handle_build_pool(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    outputs = [
        layout.selector_pool_dir,
        layout.private_registry,
        layout.pool_manifest,
        *(layout.learnware_manifest(task) for task in config.environment.tasks),
    ]
    if args.resume and layout.pool_manifest.is_file():
        manifest = read_json(layout.pool_manifest)
        if (
            isinstance(manifest, Mapping)
            and manifest.get("complete") is True
            and manifest.get("protocol_draft_hash") == config.draft_hash
        ):
            task_hashes = manifest.get("task_rkme_sha256")
            learnware_hashes = manifest.get("learnware_manifest_sha256")
            if (
                not isinstance(task_hashes, Mapping)
                or not isinstance(learnware_hashes, Mapping)
                or set(task_hashes) != set(config.environment.tasks)
                or set(learnware_hashes) != set(config.environment.tasks)
                or manifest.get("frozen_protocol_sha256")
                != sha256_file(layout.frozen_protocol)
                or manifest.get("championization_sha256")
                != sha256_file(layout.championization)
                or manifest.get("ranking_gate_sha256")
                != sha256_file(layout.reduced_unreduced_ranking)
                or manifest.get("public_pool_manifest_sha256")
                != sha256_file(layout.selector_pool_dir / "pool_manifest.json")
                or manifest.get("private_registry_sha256")
                != sha256_file(layout.private_registry)
                or any(
                    task_hashes[task] != sha256_file(layout.task_rkme(task))
                    for task in config.environment.tasks
                )
                or any(
                    learnware_hashes[task]
                    != sha256_file(layout.learnware_manifest(task))
                    for task in config.environment.tasks
                )
            ):
                raise ArtifactLayoutError("pool resume artifact digest mismatch")
            protocol = _load_frozen_protocol(layout, config)
            ranking_payload = read_json(layout.reduced_unreduced_ranking)
            if not isinstance(ranking_payload, Mapping):
                raise ArtifactLayoutError("ranking gate artifact is not an object")
            _validate_ranking_artifact(
                ranking_payload,
                config,
                layout,
                expected_protocol_id=protocol.protocol_id,
            )
            public_pool = load_public_pool(layout.selector_pool_dir)
            if (
                manifest.get("protocol_id") != protocol.protocol_id
                or public_pool.protocol_id != protocol.protocol_id
                or manifest.get("public_pool_digest")
                != sha256_json(public_pool.public_manifest())
            ):
                raise ArtifactLayoutError("pool resume protocol/public digest mismatch")
            load_private_registry(layout.private_registry, public_pool=public_pool)
            return {"resumed": True, "pool_manifest": str(layout.pool_manifest)}
        raise ArtifactLayoutError("pool resume validation failed")
    _assert_output_state(outputs, resume=args.resume)
    protocol = _load_frozen_protocol(layout, config)
    ranking_payload = read_json(layout.reduced_unreduced_ranking)
    if not isinstance(ranking_payload, Mapping):
        raise CommandFailure("reduced/unreduced ranking artifact binding failed")
    _validate_ranking_artifact(
        ranking_payload,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    task_specs: dict[str, ReducedRKME] = {}
    for task in config.environment.tasks:
        manifest = read_json(layout.task_rkme_manifest(task))
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("complete") is not True
            or manifest.get("source_task") != task
            or manifest.get("protocol_id") != protocol.protocol_id
            or manifest.get("rkme_sha256") != sha256_file(layout.task_rkme(task))
        ):
            raise CommandFailure(f"TaskSpec manifest binding failed for {task}")
        reduced = ReducedRKME.load_npz(layout.task_rkme(task))
        if (
            reduced.source_task != task
            or reduced.source_dataset_digest != manifest.get("source_dataset_digest")
            or reduced.source_dataset_manifest_digest
            != manifest.get("source_dataset_manifest_digest")
        ):
            raise CommandFailure(f"TaskSpec NPZ provenance binding failed for {task}")
        task_specs[task] = reduced
    champion_payload = read_json(layout.championization)
    championization_digest = sha256_file(layout.championization)
    inventory_digest = sha256_file(layout.policy_inventory)
    verification_digest = sha256_file(layout.bundle_verification)
    returns_digest = sha256_file(layout.championization_returns)
    if (
        not isinstance(champion_payload, Mapping)
        or champion_payload.get("complete") is not True
        or champion_payload.get("protocol_draft_hash") != config.draft_hash
        or champion_payload.get("inventory_sha256") != inventory_digest
        or champion_payload.get("verification_sha256") != verification_digest
        or champion_payload.get("returns_sha256") != returns_digest
        or champion_payload.get("environment_manifest_sha256")
        != sha256_file(layout.environment_manifest)
    ):
        raise CommandFailure("championization artifact is incomplete or incompatible")
    recomputed_champions = _recompute_championization_from_artifacts(layout, config)
    task_records = champion_payload.get("tasks")
    if not isinstance(task_records, Mapping) or set(task_records) != set(
        config.environment.tasks
    ):
        raise CommandFailure("championization task coverage mismatch")
    champions: dict[str, Any] = {}
    for task in config.environment.tasks:
        raw_record = task_records[task]
        if not isinstance(raw_record, Mapping):
            raise CommandFailure(f"champion record is not an object for {task}")
        expected = recomputed_champions.by_task[task].selected
        selected_dir = Path(str(raw_record.get("selected_bundle_dir", ""))).resolve()
        if (
            selected_dir != expected.candidate.bundle_dir.resolve()
            or str(raw_record.get("selected_bundle_digest", ""))
            != expected.candidate.bundle_digest
            or str(raw_record.get("selected_algorithm", ""))
            != expected.candidate.algorithm
            or int(raw_record.get("selected_training_seed", -1))
            != expected.candidate.training_seed
            or float(raw_record.get("selected_mean_return", float("nan")))
            != expected.mean_return
            or float(raw_record.get("selected_return_std", float("nan")))
            != expected.return_std
        ):
            raise CommandFailure(
                f"stored champion differs from recomputed fixed winner for {task}"
            )
        metadata = validate_bundle(
            selected_dir,
            expected_task=task,
            expected_algorithm=expected.candidate.algorithm,
            expected_seed=expected.candidate.training_seed,
            expected_outer=config.pool.checkpoint_outer,
            expected_environment_steps=config.pool.actual_environment_steps,
        )
        if metadata.bundle_digest != expected.candidate.bundle_digest:
            raise CommandFailure(f"selected champion bundle digest changed for {task}")
        champions[task] = metadata
    built = build_pool(task_specs, champions, protocol=protocol)
    save_public_pool(built.public_pool, layout.selector_pool_dir)
    save_private_registry(built.private_registry, layout.private_registry)
    entry_by_task = {
        built.private_registry.get(entry.opaque_id).source_task: entry
        for entry in built.public_pool.entries
    }
    learnware_digests: dict[str, str] = {}
    for task in config.environment.tasks:
        entry = entry_by_task[task]
        record = built.private_registry.get(entry.opaque_id)
        learnware_payload = {
            "schema": "policy-learnware.entry.v0",
            "learnware_id": entry.opaque_id,
            "source_task": task,
            "protocol_id": protocol.protocol_id,
            "task_spec_sha256": sha256_file(layout.task_rkme(task)),
            "task_spec_digest": entry.task_spec.task_spec_digest,
            "policy_bundle": str(record.policy_bundle),
            "policy_bundle_digest": record.policy_bundle_digest,
            "native_observation_dim": record.native_observation_dim,
            "native_action_dim": record.native_action_dim,
            "provenance": dict(record.provenance),
        }
        learnware_digests[task] = layout.publish_json(
            layout.learnware_manifest(task), learnware_payload
        )
    public_digest = sha256_json(built.public_pool.public_manifest())
    pool_manifest = {
        "schema": "policy-learnware.pool-build.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": built.public_pool.pool_id,
        "frozen_protocol_sha256": sha256_file(layout.frozen_protocol),
        "championization_sha256": championization_digest,
        "ranking_gate_sha256": sha256_file(layout.reduced_unreduced_ranking),
        "public_pool_digest": public_digest,
        "public_pool_manifest_sha256": sha256_file(
            layout.selector_pool_dir / "pool_manifest.json"
        ),
        "private_registry_sha256": sha256_file(layout.private_registry),
        "entry_count": len(built.public_pool.entries),
        "task_rkme_sha256": {
            task: sha256_file(layout.task_rkme(task))
            for task in config.environment.tasks
        },
        "learnware_manifest_sha256": learnware_digests,
        "entries": {
            task: {
                "opaque_id": entry_by_task[task].opaque_id,
                "task_spec_digest": entry_by_task[task].task_spec.task_spec_digest,
                "policy_bundle_digest": built.private_registry.get(
                    entry_by_task[task].opaque_id
                ).policy_bundle_digest,
            }
            for task in config.environment.tasks
        },
    }
    digest = layout.publish_json(layout.pool_manifest, pool_manifest)
    return {
        "pool_manifest": str(layout.pool_manifest),
        "sha256": digest,
        "public_pool_digest": public_digest,
        "entry_count": len(built.public_pool.entries),
    }


def _query_id(task: str, bank: int, episode_count: int) -> str:
    return f"bank{bank:03d}__{task}__n{episode_count:03d}"


def _retrieval_evaluator_contract() -> dict[str, Any]:
    """Bind the exact target-prefix execution path independently of TaskSpec."""

    package_root = Path(__file__).resolve().parent
    sources = {
        relative: sha256_file(package_root / relative)
        for relative in (
            "cli.py",
            "evaluation/retrieval.py",
            "evaluation/retrieval_accel.py",
            "reuse/selector.py",
            "pool/learnware.py",
            "rkme/empirical.py",
            "rkme/gaussian.py",
        )
    }
    execution = {
        "target_self_norm": "exact-jax-blockwise-nested-prefix",
        "target_source_cross": "exact-blockwise-public-supports",
        "distance": "target_norm2-2*cross+source_rkme_norm2",
        "weights": "episode-balanced",
        "tie_break": "opaque-id-lexical",
        "candidate_policy_rollout": False,
    }
    return {
        "schema": "policy-learnware.retrieval-evaluator.v0",
        "source_sha256": sources,
        "contract_sha256": sha256_json({"sources": sources, **execution}),
        "runtime": {"python": sys.version, **_runtime_package_versions()},
        **execution,
    }


def _retrieval_execution_attestation_payload(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    retrieval_payload: Mapping[str, Any],
) -> dict[str, Any]:
    queries = retrieval_payload.get("queries")
    if not isinstance(queries, list):
        raise ArtifactLayoutError("retrieval metrics have no attested queries")
    selection_digests: dict[str, str] = {}
    for raw in queries:
        if not isinstance(raw, Mapping):
            raise ArtifactLayoutError("retrieval attestation query is malformed")
        query_id = str(raw.get("query_id", ""))
        digest = str(raw.get("selection_sha256", ""))
        if not query_id or query_id in selection_digests or len(digest) != 64:
            raise ArtifactLayoutError("retrieval attestation query binding is invalid")
        selection_digests[query_id] = digest
    return {
        "schema": "policy-learnware.retrieval-execution-attestation.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": retrieval_payload.get("protocol_id"),
        "pool_id": retrieval_payload.get("pool_id"),
        "public_pool_digest": retrieval_payload.get("public_pool_digest"),
        "retrieval_metrics_sha256": sha256_file(layout.retrieval_metrics),
        "selection_sha256": selection_digests,
        "evaluator_contract": _retrieval_evaluator_contract(),
    }


def _ensure_retrieval_execution_attestation(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    retrieval_payload: Mapping[str, Any],
    *,
    create: bool,
) -> str:
    expected = _retrieval_execution_attestation_payload(
        layout, config, retrieval_payload
    )
    path = layout.retrieval_execution_attestation
    if path.is_file():
        actual = read_json(path)
        if canonicalize(actual) != canonicalize(expected):
            raise ArtifactLayoutError(
                "retrieval execution attestation differs from current source/outputs"
            )
        return sha256_file(path)
    if path.exists():
        raise ArtifactLayoutError("retrieval execution attestation is not a file")
    if not create:
        raise ArtifactLayoutError("retrieval execution attestation is missing")
    return layout.publish_json(path, expected)


def _load_verified_pool_build_manifest(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    protocol: FrozenProtocol,
    pool: Any,
) -> Mapping[str, Any]:
    manifest = read_json(layout.pool_manifest)
    public_digest = sha256_json(pool.public_manifest())
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != "policy-learnware.pool-build.v0"
        or manifest.get("complete") is not True
        or manifest.get("protocol_draft_hash") != config.draft_hash
        or manifest.get("protocol_id") != protocol.protocol_id
        or manifest.get("pool_id") != pool.pool_id
        or manifest.get("frozen_protocol_sha256")
        != sha256_file(layout.frozen_protocol)
        or manifest.get("championization_sha256")
        != sha256_file(layout.championization)
        or manifest.get("ranking_gate_sha256")
        != sha256_file(layout.reduced_unreduced_ranking)
        or manifest.get("public_pool_digest") != public_digest
        or manifest.get("public_pool_manifest_sha256")
        != sha256_file(layout.selector_pool_dir / "pool_manifest.json")
        or manifest.get("private_registry_sha256")
        != sha256_file(layout.private_registry)
    ):
        raise CommandFailure("pool build manifest is incomplete or digest-incompatible")
    ranking_payload = read_json(layout.reduced_unreduced_ranking)
    if not isinstance(ranking_payload, Mapping):
        raise CommandFailure("ranking gate artifact is not an object")
    _validate_ranking_artifact(
        ranking_payload,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    task_hashes = manifest.get("task_rkme_sha256")
    learnware_hashes = manifest.get("learnware_manifest_sha256")
    raw_entries = manifest.get("entries")
    expected_tasks = set(config.environment.tasks)
    if (
        not isinstance(task_hashes, Mapping)
        or not isinstance(learnware_hashes, Mapping)
        or not isinstance(raw_entries, Mapping)
        or set(task_hashes) != expected_tasks
        or set(learnware_hashes) != expected_tasks
        or set(raw_entries) != expected_tasks
        or any(
            task_hashes[task] != sha256_file(layout.task_rkme(task))
            or learnware_hashes[task]
            != sha256_file(layout.learnware_manifest(task))
            for task in config.environment.tasks
        )
    ):
        raise CommandFailure("pool build task artifact binding mismatch")
    public_by_id = {entry.opaque_id: entry for entry in pool.entries}
    if len(public_by_id) != len(config.environment.tasks):
        raise CommandFailure("public pool entry count differs from registered tasks")
    for task in config.environment.tasks:
        record = raw_entries[task]
        if not isinstance(record, Mapping):
            raise CommandFailure(f"pool entry mapping is invalid for {task}")
        opaque_id = str(record.get("opaque_id", ""))
        if opaque_id not in public_by_id:
            raise CommandFailure(f"pool entry mapping has unknown opaque id for {task}")
        if record.get("task_spec_digest") != public_by_id[
            opaque_id
        ].task_spec.task_spec_digest:
            raise CommandFailure(f"pool TaskSpec digest mapping mismatch for {task}")
    if {str(raw_entries[task]["opaque_id"]) for task in config.environment.tasks} != set(
        public_by_id
    ):
        raise CommandFailure("pool task-to-opaque mapping is not one-to-one")
    return manifest


def _load_verified_retrieval_artifacts(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    query_paths: Sequence[Path],
) -> Mapping[str, Any]:
    payload = read_json(layout.retrieval_metrics)
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    pool_manifest = _load_verified_pool_build_manifest(
        layout, config, protocol, pool
    )
    public_digest = sha256_json(pool.public_manifest())
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "policy-learnware.retrieval-metrics.v0"
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("protocol_id") != protocol.protocol_id
        or payload.get("pool_id") != pool.pool_id
        or payload.get("public_pool_digest") != public_digest
        or int(payload.get("target_query_banks", -1))
        != config.episodes.target_query_banks
        or list(payload.get("prefix_grid", []))
        != list(config.episodes.target_query_prefix_grid)
    ):
        raise ArtifactLayoutError("retrieval resume metrics binding mismatch")
    raw_queries = payload.get("queries")
    if not isinstance(raw_queries, list) or len(raw_queries) != len(query_paths):
        raise ArtifactLayoutError("retrieval resume query coverage mismatch")
    records: dict[str, Mapping[str, Any]] = {}
    for record in raw_queries:
        if not isinstance(record, Mapping):
            raise ArtifactLayoutError("retrieval resume query is not an object")
        query_id = str(record.get("query_id", ""))
        if not query_id or query_id in records:
            raise ArtifactLayoutError("retrieval resume has duplicate/empty query id")
        records[query_id] = record
    expected_by_task = {
        task: str(pool_manifest["entries"][task]["opaque_id"])
        for task in config.environment.tasks
    }
    pool_opaque_ids = frozenset(entry.opaque_id for entry in pool.entries)
    seed_plan = SeedPlan(config.project_seed)
    trials: list[RetrievalTrial] = []
    by_prefix: dict[int, list[RetrievalTrial]] = {
        count: [] for count in config.episodes.target_query_prefix_grid
    }
    expected_query_ids: set[str] = set()
    for bank in range(config.episodes.target_query_banks):
        for task_index, task in enumerate(config.environment.tasks):
            dataset, manifest = load_dataset_artifact(
                layout.dataset_npz("target_query", task, bank=bank),
                layout.dataset_manifest("target_query", task, bank=bank),
            )
            expected_seeds = seed_plan.episodes(
                "target_query",
                task_index,
                range(config.episodes.target_query_max_per_task),
                bank_index=bank,
            )
            if (
                manifest.task != task
                or manifest.split != "target_query"
                or manifest.protocol_draft_hash != config.draft_hash
                or tuple(dataset.reset_seeds)
                != tuple(item.reset_seed for item in expected_seeds)
                or tuple(dataset.probe_seeds)
                != tuple(item.probe_seed for item in expected_seeds)
            ):
                raise ArtifactLayoutError(
                    f"retrieval resume target dataset mismatch: {task}/{bank}"
                )
            for episode_count in config.episodes.target_query_prefix_grid:
                query_id = _query_id(task, bank, episode_count)
                expected_query_ids.add(query_id)
                record = records.get(query_id)
                selection_path = layout.selection_result(query_id)
                if record is None or not selection_path.is_file():
                    raise ArtifactLayoutError(f"retrieval resume misses {query_id}")
                prefix_digest = dataset.prefix(episode_count).digest
                if (
                    record.get("target_task") != task
                    or int(record.get("bank_index", -1)) != bank
                    or int(record.get("episode_count", -1)) != episode_count
                    or record.get("target_dataset_digest") != prefix_digest
                    or record.get("expected_opaque_id") != expected_by_task[task]
                    or record.get("selection_sha256") != sha256_file(selection_path)
                ):
                    raise ArtifactLayoutError(
                        f"retrieval resume query record mismatch: {query_id}"
                    )
                selection_payload = read_json(selection_path)
                if not isinstance(selection_payload, Mapping):
                    raise ArtifactLayoutError(f"invalid selection result: {query_id}")
                selection = SelectionResult.from_dict(selection_payload)
                trial = RetrievalTrial(expected_by_task[task], selection)
                if (
                    selection.protocol_id != protocol.protocol_id
                    or selection.pool_id != pool.pool_id
                    or selection.pool_digest != public_digest
                    or selection.target_dataset_digest != prefix_digest
                    or selection.probe_episode_count != episode_count
                    or selection.probe_steps
                    != int(dataset.episode_offsets[episode_count])
                    or frozenset(
                        item.opaque_id for item in selection.sorted_distances
                    )
                    != pool_opaque_ids
                    or len(selection.sorted_distances) != len(pool_opaque_ids)
                    or record.get("selected_opaque_id")
                    != selection.selected_opaque_id
                    or bool(record.get("correct")) != trial.correct
                ):
                    raise ArtifactLayoutError(
                        f"retrieval resume selection binding mismatch: {query_id}"
                    )
                trials.append(trial)
                by_prefix[episode_count].append(trial)
    if set(records) != expected_query_ids:
        raise ArtifactLayoutError("retrieval resume has unexpected query ids")
    expected_overall = asdict(summarize_retrieval(trials))
    expected_by_prefix = {
        str(count): asdict(summarize_retrieval(values))
        for count, values in by_prefix.items()
    }
    if (
        canonicalize(payload.get("overall")) != canonicalize(expected_overall)
        or canonicalize(payload.get("by_prefix"))
        != canonicalize(expected_by_prefix)
    ):
        raise ArtifactLayoutError("retrieval resume metric summary mismatch")
    max_prefix = max(config.episodes.target_query_prefix_grid)
    expected_gate = retrieval_gate(
        max_prefix_accuracy=float(expected_by_prefix[str(max_prefix)]["accuracy"]),
        minimum_max_prefix_accuracy=(
            config.gates.retrieval.minimum_max_prefix_accuracy
        ),
    )
    if (
        canonicalize(payload.get("gate"))
        != canonicalize(expected_gate.to_dict())
        or payload.get("gate_passed") is not expected_gate.passed
        or int(payload.get("gate_prefix_episode_count", -1)) != max_prefix
    ):
        raise ArtifactLayoutError("retrieval gate audit record mismatch")
    _require_gate_passed(
        payload,
        expected_name="exact_recurrent_retrieval",
        artifact=layout.retrieval_metrics,
    )
    if layout.retrieval_execution_attestation.exists():
        _ensure_retrieval_execution_attestation(
            layout, config, payload, create=False
        )
    return payload


def _load_partial_retrieval_selection(
    path: Path,
    *,
    query_id: str,
    task: str,
    bank: int,
    episode_count: int,
    probe_steps: int,
    target_dataset_digest: str,
    expected_opaque_id: str,
    protocol_id: str,
    pool_id: str,
    public_pool_digest: str,
    pool_opaque_ids: frozenset[str],
) -> tuple[SelectionResult, RetrievalTrial, dict[str, Any]]:
    """Load one immutable query checkpoint and validate all available bindings."""

    if not path.is_file():
        raise ArtifactLayoutError(f"retrieval checkpoint is not a file: {query_id}")
    try:
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError("selection payload is not an object")
        selection = SelectionResult.from_dict(payload)
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactLayoutError(
            f"invalid retrieval selection checkpoint: {query_id}"
        ) from error
    ranking_ids = frozenset(item.opaque_id for item in selection.sorted_distances)
    if (
        selection.protocol_id != protocol_id
        or selection.pool_id != pool_id
        or selection.pool_digest != public_pool_digest
        or selection.target_dataset_digest != target_dataset_digest
        or selection.probe_episode_count != episode_count
        or selection.probe_steps != probe_steps
        or ranking_ids != pool_opaque_ids
        or len(selection.sorted_distances) != len(pool_opaque_ids)
    ):
        raise ArtifactLayoutError(
            f"retrieval selection checkpoint binding mismatch: {query_id}"
        )
    trial = RetrievalTrial(expected_opaque_id, selection)
    record = {
        "query_id": query_id,
        "target_task": task,
        "bank_index": bank,
        "episode_count": episode_count,
        "target_dataset_digest": target_dataset_digest,
        "expected_opaque_id": expected_opaque_id,
        "selected_opaque_id": selection.selected_opaque_id,
        "correct": trial.correct,
        "selection_sha256": sha256_file(path),
    }
    return selection, trial, record


def _handle_evaluate_retrieval(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    if (shard_index is None) != (shard_count is None):
        raise CommandFailure("--shard-index and --shard-count must be provided together")
    if shard_index is not None and (
        shard_count <= 0 or shard_index < 0 or shard_index >= shard_count
    ):
        raise CommandFailure("retrieval shard coordinates are invalid")
    query_paths = [
        layout.selection_result(_query_id(task, bank, episode_count))
        for bank in range(config.episodes.target_query_banks)
        for task in config.environment.tasks
        for episode_count in config.episodes.target_query_prefix_grid
    ]
    outputs = [*query_paths, layout.retrieval_metrics]
    if shard_index is not None and layout.retrieval_metrics.exists():
        raise ArtifactExistsError(
            "retrieval is already finalized; shard workers cannot run"
        )
    if args.resume and layout.retrieval_metrics.is_file():
        payload = _load_verified_retrieval_artifacts(
            layout, config, query_paths
        )
        attestation_sha256 = _ensure_retrieval_execution_attestation(
            layout, config, payload, create=True
        )
        overall = payload["overall"]
        return {
            "resumed": True,
            "metrics": str(layout.retrieval_metrics),
            "trial_count": int(overall["trial_count"]),
            "accuracy": float(overall["accuracy"]),
            "resumed_query_count": len(query_paths),
            "computed_query_count": 0,
            "retrieval_execution_attestation_sha256": attestation_sha256,
        }
    if args.resume:
        if layout.retrieval_metrics.exists():
            raise ArtifactLayoutError(
                "retrieval metrics checkpoint exists but is not a regular file"
            )
        # Selection results are query-level immutable checkpoints.  Their
        # presence without the final metrics artifact is an expected interrupted
        # run, not a partial-output error; each one is validated below.
    else:
        _assert_output_state(outputs, resume=False)
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    if pool.protocol_id != protocol.protocol_id:
        raise CommandFailure("selector pool protocol differs from FrozenProtocol")
    pool_manifest = _load_verified_pool_build_manifest(
        layout, config, protocol, pool
    )
    raw_entries = pool_manifest["entries"]
    expected_by_task = {
        task: str(raw_entries[task]["opaque_id"]) for task in config.environment.tasks
    }
    public_pool_digest = sha256_json(pool.public_manifest())
    pool_opaque_ids = frozenset(entry.opaque_id for entry in pool.entries)
    selector: NearestSpecSelector | None = None
    schemas: Mapping[str, EnvSchema] | None = None
    canonicalizer: TransitionCanonicalizer | None = None
    encoder: TransitionSemanticEncoder | None = None
    kernel: GaussianKernel | None = None
    seed_plan = SeedPlan(config.project_seed)
    all_trials: list[RetrievalTrial] = []
    trials_by_prefix: dict[int, list[RetrievalTrial]] = {
        count: [] for count in config.episodes.target_query_prefix_grid
    }
    query_records: list[dict[str, Any]] = []
    resumed_query_count = 0
    computed_query_count = 0
    for bank in range(config.episodes.target_query_banks):
        for task_index, task in enumerate(config.environment.tasks):
            group_index = bank * len(config.environment.tasks) + task_index
            if shard_index is not None and group_index % shard_count != shard_index:
                continue
            dataset, manifest = load_dataset_artifact(
                layout.dataset_npz("target_query", task, bank=bank),
                layout.dataset_manifest("target_query", task, bank=bank),
            )
            if (
                manifest.task != task
                or manifest.split != "target_query"
                or manifest.protocol_draft_hash != config.draft_hash
            ):
                raise CommandFailure(f"target query manifest binding mismatch: {task}/{bank}")
            expected_seeds = seed_plan.episodes(
                "target_query",
                task_index,
                range(config.episodes.target_query_max_per_task),
                bank_index=bank,
            )
            if tuple(dataset.reset_seeds) != tuple(
                item.reset_seed for item in expected_seeds
            ) or tuple(dataset.probe_seeds) != tuple(
                item.probe_seed for item in expected_seeds
            ):
                raise CommandFailure(f"target query seeds mismatch: {task}/{bank}")

            prefix_info: list[tuple[int, int, str, str, Path]] = []
            for episode_count in config.episodes.target_query_prefix_grid:
                transition_stop = int(dataset.episode_offsets[episode_count])
                prefix = dataset.prefix(episode_count)
                query_id = _query_id(task, bank, episode_count)
                prefix_info.append(
                    (
                        episode_count,
                        transition_stop,
                        prefix.digest,
                        query_id,
                        layout.selection_result(query_id),
                    )
                )

            group_results: dict[
                int, tuple[RetrievalTrial, dict[str, Any]]
            ] = {}
            missing: list[tuple[int, int, str, str, Path]] = []
            for (
                episode_count,
                transition_stop,
                prefix_digest,
                query_id,
                selection_path,
            ) in prefix_info:
                if selection_path.exists():
                    if not args.resume:
                        raise ArtifactExistsError(
                            f"refusing to overwrite immutable output: {selection_path}"
                        )
                    _, trial, record = _load_partial_retrieval_selection(
                        selection_path,
                        query_id=query_id,
                        task=task,
                        bank=bank,
                        episode_count=episode_count,
                        probe_steps=transition_stop,
                        target_dataset_digest=prefix_digest,
                        expected_opaque_id=expected_by_task[task],
                        protocol_id=protocol.protocol_id,
                        pool_id=pool.pool_id,
                        public_pool_digest=public_pool_digest,
                        pool_opaque_ids=pool_opaque_ids,
                    )
                    group_results[episode_count] = (trial, record)
                    resumed_query_count += 1
                else:
                    missing.append(
                        (
                            episode_count,
                            transition_stop,
                            prefix_digest,
                            query_id,
                            selection_path,
                        )
                    )

            if missing:
                if canonicalizer is None:
                    schemas = _load_env_schemas(layout, config)
                    stats = NormalizationStats.load_npz(layout.normalization)
                    canonicalizer = TransitionCanonicalizer(
                        stats=stats,
                        max_action_dim=config.environment.max_action_dim,
                    )
                    checkpoint = EncoderCheckpoint.load(
                        layout.encoder_checkpoint,
                        read_json(layout.encoder_config),
                    )
                    encoder = TransitionSemanticEncoder(checkpoint)
                    kernel = GaussianKernel.load_json(layout.kernel)
                    selector = NearestSpecSelector(
                        pool,
                        negative_tolerance=config.selector.negative_tolerance,
                    )
                assert schemas is not None
                assert encoder is not None
                assert kernel is not None
                assert selector is not None
                packed = canonicalizer.pack(dataset, schemas[task])
                all_points = encoder.encode(packed.packed)
                if all_points.ndim != 2 or all_points.shape[1] != pool.latent_dim:
                    raise CommandFailure(
                        "target latent dimension differs from selector pool"
                    )
                if not np.isclose(
                    kernel.bandwidth,
                    pool.kernel_bandwidth,
                    rtol=1.0e-12,
                    atol=0.0,
                ):
                    raise CommandFailure(
                        "target kernel bandwidth differs from selector pool"
                    )

                missing_counts = tuple(item[0] for item in missing)
                maximum_missing_count = max(missing_counts)
                maximum_transition_stop = int(
                    dataset.episode_offsets[maximum_missing_count]
                )
                exact_norms = nested_prefix_self_kernel_sums_jax(
                    all_points[:maximum_transition_stop],
                    dataset.episode_offsets[: maximum_missing_count + 1],
                    missing_counts,
                    kernel,
                )

            for (
                episode_count,
                transition_stop,
                prefix_digest,
                query_id,
                selection_path,
            ) in missing:
                prefix_weights = episode_balanced_weights(
                    dataset.episode_offsets[: episode_count + 1]
                )
                cross_terms = target_source_cross_terms(
                    all_points[:transition_stop], prefix_weights, pool
                )
                selection = selector.select_from_precomputed_terms(
                    target_empirical_norm2=exact_norms[episode_count],
                    target_source_cross=cross_terms,
                    target_dataset_digest=prefix_digest,
                    probe_episode_count=episode_count,
                    probe_steps=transition_stop,
                )
                layout.publish_json(selection_path, selection.to_dict())
                _, trial, record = _load_partial_retrieval_selection(
                    selection_path,
                    query_id=query_id,
                    task=task,
                    bank=bank,
                    episode_count=episode_count,
                    probe_steps=transition_stop,
                    target_dataset_digest=prefix_digest,
                    expected_opaque_id=expected_by_task[task],
                    protocol_id=protocol.protocol_id,
                    pool_id=pool.pool_id,
                    public_pool_digest=public_pool_digest,
                    pool_opaque_ids=pool_opaque_ids,
                )
                group_results[episode_count] = (trial, record)
                computed_query_count += 1

            for episode_count, *_ in prefix_info:
                trial, record = group_results[episode_count]
                all_trials.append(trial)
                trials_by_prefix[episode_count].append(trial)
                query_records.append(record)
    if shard_index is not None:
        return {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "resumed_query_count": resumed_query_count,
            "computed_query_count": computed_query_count,
            "finalized": False,
        }
    overall = summarize_retrieval(all_trials)
    by_prefix = {
        str(count): asdict(summarize_retrieval(trials))
        for count, trials in trials_by_prefix.items()
    }
    max_prefix = max(config.episodes.target_query_prefix_grid)
    retrieval_decision = retrieval_gate(
        max_prefix_accuracy=float(by_prefix[str(max_prefix)]["accuracy"]),
        minimum_max_prefix_accuracy=(
            config.gates.retrieval.minimum_max_prefix_accuracy
        ),
    )
    metrics_payload = {
        "schema": "policy-learnware.retrieval-metrics.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": pool.pool_id,
        "public_pool_digest": public_pool_digest,
        "target_query_banks": config.episodes.target_query_banks,
        "prefix_grid": list(config.episodes.target_query_prefix_grid),
        "overall": asdict(overall),
        "by_prefix": by_prefix,
        "gate_prefix_episode_count": max_prefix,
        "gate": retrieval_decision.to_dict(),
        "gate_passed": retrieval_decision.passed,
        "queries": query_records,
    }
    digest = layout.publish_json(layout.retrieval_metrics, metrics_payload)
    attestation_sha256 = _ensure_retrieval_execution_attestation(
        layout, config, metrics_payload, create=True
    )
    _require_gate_passed(
        metrics_payload,
        expected_name="exact_recurrent_retrieval",
        artifact=layout.retrieval_metrics,
    )
    return {
        "metrics": str(layout.retrieval_metrics),
        "sha256": digest,
        "trial_count": overall.trial_count,
        "accuracy": overall.accuracy,
        "gate_passed": retrieval_decision.passed,
        "resumed_query_count": resumed_query_count,
        "computed_query_count": computed_query_count,
        "retrieval_execution_attestation_sha256": attestation_sha256,
    }


def _evaluate_frozen_policy_returns_accelerated(
    policy: Any,
    *,
    task: str,
    reset_seeds: Sequence[int],
    policy_seeds: Sequence[int],
    config: ProtocolDraft,
    expected_schema: EnvSchema,
) -> tuple[float, ...]:
    if len(reset_seeds) != len(policy_seeds) or not reset_seeds:
        raise CommandFailure("final-return seed vectors are empty or misaligned")
    adapter = make_env_adapter(task, config, jit=True)
    if adapter.schema.digest != expected_schema.digest:
        raise CommandFailure(f"live environment schema drift for {task}")
    try:
        return evaluate_frozen_policy_returns_batched(
            policy,
            adapter.environment,
            reset_seeds=reset_seeds,
            policy_seeds=policy_seeds,
            horizon=adapter.schema.horizon,
            observation_dim=adapter.schema.observation_dim,
            action_dim=adapter.schema.action_dim,
        )
    except RuntimeError as error:
        raise CommandFailure(
            f"batched frozen-policy evaluation failed for {task}: {error}"
        ) from error


def _evaluate_frozen_policy_returns(
    policy: Any,
    *,
    task: str,
    reset_seeds: Sequence[int],
    policy_seeds: Sequence[int],
    config: ProtocolDraft,
    expected_schema: EnvSchema,
) -> tuple[float, ...]:
    """Run the original scalar evaluator used by final deployment metrics.

    Championization has its own explicitly versioned accelerator-resident
    evaluator above.  Keeping deployment on this scalar path prevents an
    orchestration optimization from silently changing the separately gated
    final-return evaluation contract.
    """

    if len(reset_seeds) != len(policy_seeds) or not reset_seeds:
        raise CommandFailure("final-return seed vectors are empty or misaligned")
    adapter = make_env_adapter(task, config, jit=True)
    if adapter.schema.digest != expected_schema.digest:
        raise CommandFailure(f"live environment schema drift for {task}")
    try:
        import jax
    except ImportError as error:  # pragma: no cover - production dependency gate
        raise CommandFailure("scalar frozen-policy evaluation requires JAX") from error

    random = jax.random
    episode_returns: list[float] = []
    for reset_seed, policy_seed in zip(reset_seeds, policy_seeds, strict=True):
        state, observation = adapter.reset(int(reset_seed))
        key = (
            random.key(int(policy_seed))
            if hasattr(random, "key")
            else random.PRNGKey(int(policy_seed))  # pragma: no cover
        )
        episode_return = 0.0
        for step_index in range(adapter.schema.horizon):
            action, key = policy.act(observation, key, deterministic=True)
            action_array = np.asarray(jax.device_get(action), dtype=np.float32)
            if action_array.shape != (adapter.schema.action_dim,):
                raise CommandFailure(
                    f"policy action shape drift for {task}: {action_array.shape}"
                )
            if not np.all(np.isfinite(action_array)):
                raise CommandFailure(f"policy emitted non-finite action for {task}")
            state, result = adapter.step(state, action_array)
            episode_return += float(result.reward)
            observation = result.observation
            if result.terminated or result.truncated:
                if step_index + 1 < adapter.schema.horizon:
                    raise CommandFailure(
                        f"environment ended before the registered horizon for {task}"
                    )
                break
        if not np.isfinite(episode_return):
            raise CommandFailure(f"policy emitted non-finite return for {task}")
        episode_returns.append(episode_return)
    return tuple(episode_returns)


def _deployment_evaluator_contract() -> dict[str, Any]:
    """Version the accelerated selected-policy rollout independently."""

    package_root = Path(__file__).resolve().parent
    sources = {
        relative: sha256_file(package_root / relative)
        for relative in (
            "policy/evaluate.py",
            "policy/loader.py",
            "policy/parity.py",
            "envs/mujoco_playground.py",
        )
    }
    execution = {
        "execution": "jax-jit-lax-map-lax-scan",
        "policy_deterministic_argument": True,
        "action_transform": "tanh(raw_action)",
        "return_accumulation": "host-float64-left-to-right",
        "cache_unit": "unique-target-task-selected-opaque-id",
        "fallback": "none",
    }
    return {
        "schema": "policy-learnware.deployment-evaluator.v0",
        "source_sha256": sources,
        "contract_sha256": sha256_json({"sources": sources, **execution}),
        "runtime": {"python": sys.version, **_runtime_package_versions()},
        **execution,
    }


def _deployment_seed_vectors(
    task: str, config: ProtocolDraft
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    task_index = config.environment.tasks.index(task)
    count = config.episodes.final_return_per_task
    plan = SeedPlan(config.project_seed)
    return (
        tuple(
            plan.derive(
                "final_return", task_index, index, stream="environment_reset"
            )
            for index in range(count)
        ),
        tuple(
            plan.derive("final_return", task_index, index, stream="policy_action")
            for index in range(count)
        ),
    )


def _deployment_pair_plan(
    raw_queries: Sequence[Mapping[str, Any]], config: ProtocolDraft
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for raw in raw_queries:
        task = str(raw.get("target_task", ""))
        opaque_id = str(raw.get("selected_opaque_id", ""))
        query_id = str(raw.get("query_id", ""))
        selection_sha256 = str(raw.get("selection_sha256", ""))
        if (
            task not in config.environment.tasks
            or not opaque_id
            or not query_id
            or len(selection_sha256) != 64
        ):
            raise CommandFailure("retrieval query cannot form a deployment pair")
        grouped.setdefault((task, opaque_id), []).append(
            {"query_id": query_id, "selection_sha256": selection_sha256}
        )
    task_order = {task: index for index, task in enumerate(config.environment.tasks)}
    plan: list[dict[str, Any]] = []
    for task, opaque_id in sorted(
        grouped, key=lambda item: (task_order[item[0]], item[1])
    ):
        bindings = sorted(grouped[(task, opaque_id)], key=lambda item: item["query_id"])
        pair_id = "dep-" + sha256_json(
            {"target_task": task, "selected_opaque_id": opaque_id}
        )[:20]
        binding = {
            "target_task": task,
            "selected_opaque_id": opaque_id,
            "queries": bindings,
        }
        plan.append(
            {
                **binding,
                "pair_id": pair_id,
                "pair_plan_digest": sha256_json(binding),
            }
        )
    if not plan:
        raise CommandFailure("deployment pair plan is empty")
    return tuple(plan)


def _validate_deployment_pair_checkpoint(
    payload: Any,
    *,
    item: Mapping[str, Any],
    config: ProtocolDraft,
    protocol: FrozenProtocol,
    pool: Any,
    registry: Any,
    schemas: Mapping[str, EnvSchema],
    evaluator_contract: Mapping[str, Any],
    layout: ArtifactLayout,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArtifactLayoutError("deployment pair checkpoint is not an object")
    task = str(item["target_task"])
    opaque_id = str(item["selected_opaque_id"])
    record = registry.get(opaque_id)
    reset_seeds, policy_seeds = _deployment_seed_vectors(task, config)
    if (
        payload.get("schema") != "policy-learnware.deployment-pair-evaluation.v0"
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("protocol_id") != protocol.protocol_id
        or payload.get("pool_id") != pool.pool_id
        or payload.get("public_pool_digest") != sha256_json(pool.public_manifest())
        or payload.get("private_registry_sha256")
        != sha256_file(layout.private_registry)
        or payload.get("retrieval_metrics_sha256")
        != sha256_file(layout.retrieval_metrics)
        or payload.get("environment_manifest_sha256")
        != sha256_file(layout.environment_manifest)
        or payload.get("pair_id") != item["pair_id"]
        or payload.get("pair_plan_digest") != item["pair_plan_digest"]
        or canonicalize(payload.get("queries"))
        != canonicalize(item["queries"])
        or payload.get("target_task") != task
        or payload.get("target_schema_digest") != schemas[task].digest
        or payload.get("selected_opaque_id") != opaque_id
        or payload.get("policy_bundle_digest") != record.policy_bundle_digest
        or int(payload.get("final_return_episode_count", -1))
        != config.episodes.final_return_per_task
        or tuple(payload.get("reset_seeds", ())) != reset_seeds
        or tuple(payload.get("policy_seeds", ())) != policy_seeds
        or canonicalize(payload.get("evaluator_contract"))
        != canonicalize(evaluator_contract)
        or payload.get("fallback") != "none"
        or payload.get("status") not in {"success", "deployment_failure"}
    ):
        raise ArtifactLayoutError(
            f"deployment pair checkpoint binding mismatch: {item['pair_id']}"
        )
    returns = tuple(float(value) for value in payload.get("episode_returns", ()))
    runtime = float(payload.get("runtime_seconds", -1.0))
    if not np.isfinite(runtime) or runtime < 0.0:
        raise ArtifactLayoutError("deployment pair runtime is invalid")
    if payload["status"] == "success":
        golden = payload.get("golden_parity")
        compiled = payload.get("compiled_parity")
        device = payload.get("device")
        if (
            len(returns) != config.episodes.final_return_per_task
            or any(not np.isfinite(value) for value in returns)
            or payload.get("deployment_failure") is not None
            or float(payload.get("mean_return")) != fmean(returns)
            or float(payload.get("return_std")) != pstdev(returns)
            or not isinstance(golden, Mapping)
            or golden.get("passed") is not True
            or float(golden.get("atol", -1.0)) != config.policy.parity_atol
            or float(golden.get("rtol", -1.0)) != config.policy.parity_rtol
            or int(golden.get("sample_count", 0)) != 8
            or golden.get("raw_checked") is not True
            or not np.isfinite(float(golden.get("raw_max_abs_error", float("nan"))))
            or not np.isfinite(
                float(golden.get("environment_max_abs_error", float("nan")))
            )
            or not isinstance(compiled, Mapping)
            or compiled.get("passed") is not True
            or compiled.get("next_keys_equal") is not True
            or float(compiled.get("atol", -1.0)) != config.policy.parity_atol
            or float(compiled.get("rtol", -1.0)) != config.policy.parity_rtol
            or int(compiled.get("sample_count", 0)) != 2
            or not np.isfinite(
                float(compiled.get("max_abs_error", float("nan")))
            )
            or not isinstance(device, Mapping)
            or device.get("platform") != "gpu"
            or not device.get("device_kind")
        ):
            raise ArtifactLayoutError("successful deployment pair is inconsistent")
    elif (
        returns
        or not payload.get("deployment_failure")
        or payload.get("mean_return") is not None
        or payload.get("return_std") is not None
    ):
        raise ArtifactLayoutError("failed deployment pair is inconsistent")
    return payload


def _evaluate_deployment_pair(
    *,
    item: Mapping[str, Any],
    pair_index: int,
    shard_index: int,
    shard_count: int,
    config: ProtocolDraft,
    protocol: FrozenProtocol,
    pool: Any,
    registry: Any,
    schemas: Mapping[str, EnvSchema],
    evaluator_contract: Mapping[str, Any],
    layout: ArtifactLayout,
    resume: bool,
) -> tuple[Mapping[str, Any], bool]:
    destination = layout.deployment_pair_evaluation(str(item["pair_id"]))
    if destination.is_file():
        if not resume:
            raise ArtifactExistsError(f"refusing to overwrite artifact: {destination}")
        return (
            _validate_deployment_pair_checkpoint(
                read_json(destination),
                item=item,
                config=config,
                protocol=protocol,
                pool=pool,
                registry=registry,
                schemas=schemas,
                evaluator_contract=evaluator_contract,
                layout=layout,
            ),
            True,
        )
    if destination.exists():
        raise ArtifactLayoutError(f"deployment pair checkpoint is not a file: {destination}")

    try:
        import jax
    except ImportError as error:  # pragma: no cover - production dependency gate
        raise CommandFailure("deployment pair evaluation requires JAX") from error
    available_devices = jax.devices()
    if not available_devices or available_devices[0].platform != "gpu":
        raise CommandFailure(
            "deployment pair evaluation requires a visible GPU; "
            "no checkpoint was published"
        )
    execution_device = available_devices[0]

    task = str(item["target_task"])
    opaque_id = str(item["selected_opaque_id"])
    record = registry.get(opaque_id)
    schema = schemas[task]
    reset_seeds, policy_seeds = _deployment_seed_vectors(task, config)
    started = time.monotonic()
    print(
        f"[deployment shard {shard_index + 1}/{shard_count}] "
        f"start {pair_index + 1}: {task}/{opaque_id}",
        file=sys.stderr,
        flush=True,
    )
    status = "deployment_failure"
    failure: str | None = None
    returns: tuple[float, ...] = ()
    golden_payload: dict[str, Any] | None = None
    compiled_payload: dict[str, Any] | None = None
    device_payload: dict[str, Any] | None = None
    if (
        schema.observation_dim != record.native_observation_dim
        or schema.action_dim != record.native_action_dim
    ):
        failure = "incompatible_native_schema"
    else:
        try:
            metadata = validate_bundle(
                record.policy_bundle, expected_task=record.source_task
            )
            if metadata.bundle_digest != record.policy_bundle_digest:
                raise ValueError("registered policy bundle digest has changed")
            if (
                metadata.observation_dim != record.native_observation_dim
                or metadata.action_dim != record.native_action_dim
            ):
                raise ValueError("registered policy dimensions have changed")
            policy = load_policy(metadata, fpo_root=config.runtime.fpo_root)
            golden = verify_golden_parity(
                policy,
                metadata,
                atol=config.policy.parity_atol,
                rtol=config.policy.parity_rtol,
            )
            golden_payload = asdict(golden)
            if not golden.passed:
                raise ValueError("selected policy failed golden parity")
            with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
                compiled = verify_compiled_policy_parity(
                    policy,
                    np.asarray(archive["observation"]),
                    np.asarray(archive["prng_key_data"]),
                    atol=config.policy.parity_atol,
                    rtol=config.policy.parity_rtol,
                )
            compiled_payload = asdict(compiled)
            if not compiled.passed:
                raise ValueError("selected policy failed compiled evaluator parity")
        except (BundleValidationError, ValueError) as error:
            failure = f"policy_pair_validation_failed:{type(error).__name__}"
            returns = ()
        else:
            # Runtime/backend/resource failures are intentionally not converted
            # into immutable deployment failures.  The worker exits non-zero
            # without publishing this pair, so --resume can retry it.
            returns = _evaluate_frozen_policy_returns_accelerated(
                policy,
                task=task,
                reset_seeds=reset_seeds,
                policy_seeds=policy_seeds,
                config=config,
                expected_schema=schema,
            )
            if (
                len(returns) != config.episodes.final_return_per_task
                or any(not np.isfinite(value) for value in returns)
            ):
                raise CommandFailure(
                    "accelerated deployment returned invalid episodes"
                )
            device_payload = {
                "platform": str(execution_device.platform),
                "device_kind": str(execution_device.device_kind),
            }
            status = "success"
    elapsed = time.monotonic() - started
    payload = {
        "schema": "policy-learnware.deployment-pair-evaluation.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": pool.pool_id,
        "public_pool_digest": sha256_json(pool.public_manifest()),
        "private_registry_sha256": sha256_file(layout.private_registry),
        "retrieval_metrics_sha256": sha256_file(layout.retrieval_metrics),
        "environment_manifest_sha256": sha256_file(layout.environment_manifest),
        "pair_id": item["pair_id"],
        "pair_plan_digest": item["pair_plan_digest"],
        "queries": item["queries"],
        "target_task": task,
        "target_schema_digest": schema.digest,
        "selected_opaque_id": opaque_id,
        "policy_bundle_digest": record.policy_bundle_digest,
        "final_return_episode_count": config.episodes.final_return_per_task,
        "reset_seeds": list(reset_seeds),
        "policy_seeds": list(policy_seeds),
        "status": status,
        "deployment_failure": failure,
        "episode_returns": list(returns),
        "mean_return": fmean(returns) if returns else None,
        "return_std": pstdev(returns) if returns else None,
        "runtime_seconds": elapsed,
        "golden_parity": golden_payload,
        "compiled_parity": compiled_payload,
        "device": device_payload,
        "evaluator_contract": canonicalize(evaluator_contract),
        "fallback": "none",
    }
    layout.publish_json(destination, payload)
    validated = _validate_deployment_pair_checkpoint(
        payload,
        item=item,
        config=config,
        protocol=protocol,
        pool=pool,
        registry=registry,
        schemas=schemas,
        evaluator_contract=evaluator_contract,
        layout=layout,
    )
    print(
        f"[deployment shard {shard_index + 1}/{shard_count}] "
        f"done {pair_index + 1}: status={status} seconds={elapsed:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return validated, False


def _validate_deployment_resume(
    layout: ArtifactLayout,
    config: ProtocolDraft,
    retrieval_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = read_json(layout.deployment_metrics)
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    _load_verified_pool_build_manifest(layout, config, protocol, pool)
    registry = load_private_registry(layout.private_registry, public_pool=pool)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "policy-learnware.deployment-metrics.v0"
        or payload.get("complete") is not True
        or payload.get("protocol_draft_hash") != config.draft_hash
        or payload.get("protocol_id") != protocol.protocol_id
        or payload.get("pool_id") != pool.pool_id
        or payload.get("public_pool_digest")
        != sha256_json(pool.public_manifest())
        or payload.get("private_registry_sha256")
        != sha256_file(layout.private_registry)
        or payload.get("retrieval_metrics_sha256")
        != sha256_file(layout.retrieval_metrics)
        or int(payload.get("final_return_episode_count", -1))
        != config.episodes.final_return_per_task
        or payload.get("evaluation_mode")
        != "unique-target-task-selected-policy-pair"
        or canonicalize(payload.get("evaluator_contract"))
        != canonicalize(_deployment_evaluator_contract())
        or payload.get("selected_only") is not True
        or payload.get("fallback") != "none"
    ):
        raise ArtifactLayoutError("deployment resume metrics binding mismatch")
    raw_retrieval_queries = retrieval_payload.get("queries")
    raw_records = payload.get("queries")
    if (
        not isinstance(raw_retrieval_queries, list)
        or not isinstance(raw_records, list)
        or len(raw_records) != len(raw_retrieval_queries)
    ):
        raise ArtifactLayoutError("deployment resume query coverage mismatch")
    typed_retrieval_queries: list[Mapping[str, Any]] = []
    for raw in raw_retrieval_queries:
        if not isinstance(raw, Mapping):
            raise ArtifactLayoutError("retrieval query is not an object")
        typed_retrieval_queries.append(raw)
    schemas = _load_env_schemas(layout, config)
    pair_plan = _deployment_pair_plan(typed_retrieval_queries, config)
    raw_pair_digests = payload.get("pair_evaluations")
    if (
        int(payload.get("unique_pair_count", -1)) != len(pair_plan)
        or not isinstance(raw_pair_digests, Mapping)
        or set(raw_pair_digests) != {str(item["pair_id"]) for item in pair_plan}
    ):
        raise ArtifactLayoutError("deployment pair coverage mismatch")
    pair_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    pair_payload_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in pair_plan:
        pair_id = str(item["pair_id"])
        pair_path = layout.deployment_pair_evaluation(pair_id)
        if (
            not pair_path.is_file()
            or raw_pair_digests[pair_id] != sha256_file(pair_path)
        ):
            raise ArtifactLayoutError(f"deployment pair digest mismatch: {pair_id}")
        pair_payload = _validate_deployment_pair_checkpoint(
            read_json(pair_path),
            item=item,
            config=config,
            protocol=protocol,
            pool=pool,
            registry=registry,
            schemas=schemas,
            evaluator_contract=_deployment_evaluator_contract(),
            layout=layout,
        )
        pair_bindings[(str(item["target_task"]), str(item["selected_opaque_id"]))] = (
            pair_id,
            str(raw_pair_digests[pair_id]),
        )
        pair_payload_by_key[
            (str(item["target_task"]), str(item["selected_opaque_id"]))
        ] = pair_payload

    records: dict[str, Mapping[str, Any]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise ArtifactLayoutError("deployment resume query is not an object")
        query_id = str(record.get("query_id", ""))
        if not query_id or query_id in records:
            raise ArtifactLayoutError("deployment resume has duplicate/empty query id")
        records[query_id] = record
    outcomes: list[DeploymentResult] = []
    expected_query_ids: set[str] = set()
    for raw in typed_retrieval_queries:
        query_id = str(raw["query_id"])
        task = str(raw["target_task"])
        expected_query_ids.add(query_id)
        record = records.get(query_id)
        result_path = layout.deployment_result(query_id)
        selection_path = layout.selection_result(query_id)
        if record is None or not result_path.is_file():
            raise ArtifactLayoutError(f"deployment resume misses {query_id}")
        if (
            record.get("target_task") != task
            or record.get("selected_opaque_id") != raw.get("selected_opaque_id")
            or bool(record.get("correct_retrieval")) != bool(raw.get("correct"))
            or record.get("deployment_sha256") != sha256_file(result_path)
            or (
                record.get("pair_id"), record.get("pair_evaluation_sha256")
            )
            != pair_bindings[(task, str(raw.get("selected_opaque_id")))]
        ):
            raise ArtifactLayoutError(
                f"deployment resume query record mismatch: {query_id}"
            )
        result_payload = read_json(result_path)
        selection_payload = read_json(selection_path)
        if not isinstance(result_payload, Mapping) or not isinstance(
            selection_payload, Mapping
        ):
            raise ArtifactLayoutError(f"deployment/selection artifact malformed: {query_id}")
        outcome = DeploymentResult.from_dict(result_payload)
        selection = SelectionResult.from_dict(selection_payload)
        expected_reset, expected_policy = _deployment_seed_vectors(task, config)
        private_record = registry.get(selection.selected_opaque_id)
        pair_payload = pair_payload_by_key[(task, selection.selected_opaque_id)]
        pair_returns = tuple(float(value) for value in pair_payload["episode_returns"])
        if (
            outcome.selection_id != selection.selection_id
            or outcome.protocol_id != protocol.protocol_id
            or outcome.selected_opaque_id != selection.selected_opaque_id
            or outcome.target_observation_dim != schemas[task].observation_dim
            or outcome.target_action_dim != schemas[task].action_dim
            or outcome.policy_observation_dim != private_record.native_observation_dim
            or outcome.policy_action_dim != private_record.native_action_dim
            or outcome.evaluation_reset_seeds != expected_reset
            or outcome.evaluation_policy_seeds != expected_policy
            or record.get("status") != outcome.status
            or record.get("deployment_failure") != outcome.deployment_failure
            or record.get("mean_return") != outcome.mean_return
            or outcome.status != pair_payload["status"]
            or outcome.deployment_failure != pair_payload["deployment_failure"]
            or outcome.episode_returns != pair_returns
            or outcome.mean_return != pair_payload["mean_return"]
            or outcome.return_std != pair_payload["return_std"]
            or outcome.runtime_seconds != pair_payload["runtime_seconds"]
        ):
            raise ArtifactLayoutError(
                f"deployment resume result binding mismatch: {query_id}"
            )
        outcomes.append(outcome)
    if set(records) != expected_query_ids:
        raise ArtifactLayoutError("deployment resume has unexpected query ids")
    expected_summary = asdict(summarize_deployments(outcomes))
    if canonicalize(payload.get("summary")) != canonicalize(expected_summary):
        raise ArtifactLayoutError("deployment resume metric summary mismatch")
    correct_outcomes = [
        outcome
        for outcome, raw in zip(outcomes, typed_retrieval_queries, strict=True)
        if bool(raw["correct"])
    ]
    correct_count = len(correct_outcomes)
    correct_deployable_count = sum(outcome.deployable for outcome in correct_outcomes)
    correct_deployability_rate = (
        correct_deployable_count / correct_count if correct_count else None
    )
    expected_correct_summary = {
        "query_count": correct_count,
        "deployable_count": correct_deployable_count,
        "deployability_rate": correct_deployability_rate,
    }
    expected_gate = deployment_gate(
        correct_retrieval_count=correct_count,
        correct_retrieval_deployability_rate=correct_deployability_rate,
        minimum_correct_retrieval_deployability_rate=(
            config.gates.deployment.minimum_correct_retrieval_deployability_rate
        ),
    )
    if (
        canonicalize(payload.get("correct_retrieval_deployment"))
        != canonicalize(expected_correct_summary)
        or canonicalize(payload.get("gate"))
        != canonicalize(expected_gate.to_dict())
        or payload.get("gate_passed") is not expected_gate.passed
    ):
        raise ArtifactLayoutError("deployment gate audit record mismatch")
    _require_gate_passed(
        payload,
        expected_name="selected_policy_deployment",
        artifact=layout.deployment_metrics,
    )
    return payload


def _handle_evaluate_deployment(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    if (shard_index is None) != (shard_count is None):
        raise CommandFailure("--shard-index and --shard-count must be provided together")
    if shard_index is not None and (
        shard_count <= 0 or shard_index < 0 or shard_index >= shard_count
    ):
        raise CommandFailure("deployment shard coordinates are invalid")
    shard_mode = shard_index is not None
    selection_paths = [
        layout.selection_result(_query_id(task, bank, episode_count))
        for bank in range(config.episodes.target_query_banks)
        for task in config.environment.tasks
        for episode_count in config.episodes.target_query_prefix_grid
    ]
    retrieval_payload = _load_verified_retrieval_artifacts(
        layout, config, selection_paths
    )
    raw_queries = (
        retrieval_payload.get("queries")
        if isinstance(retrieval_payload, Mapping)
        else None
    )
    if not isinstance(raw_queries, list) or not raw_queries:
        raise CommandFailure("retrieval metrics have no query records")
    deployment_paths = [
        layout.deployment_result(str(record["query_id"])) for record in raw_queries
    ]
    outputs = [*deployment_paths, layout.deployment_metrics]
    if args.resume and layout.deployment_metrics.is_file() and not shard_mode:
        _validate_deployment_resume(layout, config, retrieval_payload)
        return {"resumed": True, "metrics": str(layout.deployment_metrics)}
    if shard_mode and layout.deployment_metrics.exists():
        raise ArtifactExistsError(
            "deployment is already finalized; shard workers cannot run"
        )
    if not args.resume:
        _assert_output_state(outputs, resume=False)
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    registry = load_private_registry(
        layout.private_registry, public_pool=pool
    )
    schemas = _load_env_schemas(layout, config)
    evaluator_contract = _deployment_evaluator_contract()
    pair_plan = _deployment_pair_plan(raw_queries, config)
    evaluated_pair_count = 0
    resumed_pair_count = 0
    pair_payloads: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
    effective_shard_index = 0 if shard_index is None else shard_index
    effective_shard_count = 1 if shard_count is None else shard_count
    for pair_index, item in enumerate(pair_plan):
        if shard_mode and pair_index % shard_count != shard_index:
            continue
        pair_payload, resumed = _evaluate_deployment_pair(
            item=item,
            pair_index=pair_index,
            shard_index=effective_shard_index,
            shard_count=effective_shard_count,
            config=config,
            protocol=protocol,
            pool=pool,
            registry=registry,
            schemas=schemas,
            evaluator_contract=evaluator_contract,
            layout=layout,
            resume=args.resume,
        )
        pair_path = layout.deployment_pair_evaluation(str(item["pair_id"]))
        pair_payloads[(str(item["target_task"]), str(item["selected_opaque_id"]))] = (
            pair_payload,
            sha256_file(pair_path),
        )
        if resumed:
            resumed_pair_count += 1
        else:
            evaluated_pair_count += 1
        if not resumed:
            try:
                import jax

                jax.clear_caches()
            except ImportError:  # pragma: no cover
                pass
            gc.collect()
    if shard_mode:
        return {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "unique_pair_count": len(pair_plan),
            "evaluated_pair_count": evaluated_pair_count,
            "resumed_pair_count": resumed_pair_count,
            "finalized": False,
        }

    outcomes: list[DeploymentResult] = []
    records: list[dict[str, Any]] = []

    for raw in raw_queries:
        if not isinstance(raw, Mapping):
            raise CommandFailure("retrieval query record is not an object")
        query_id = str(raw["query_id"])
        task = str(raw["target_task"])
        if task not in schemas:
            raise CommandFailure(f"unknown deployment target task: {task}")
        selection_path = layout.selection_result(query_id)
        if sha256_file(selection_path) != raw.get("selection_sha256"):
            raise CommandFailure(f"selection artifact digest mismatch: {query_id}")
        selection_payload = read_json(selection_path)
        if not isinstance(selection_payload, Mapping):
            raise CommandFailure(f"invalid selection artifact: {query_id}")
        selection = SelectionResult.from_dict(selection_payload)
        if (
            selection.protocol_id != protocol.protocol_id
            or selection.pool_id != registry.pool_id
            or selection.pool_digest != registry.pool_digest
            or selection.selected_opaque_id != raw["selected_opaque_id"]
        ):
            raise CommandFailure(f"selection protocol mismatch: {query_id}")
        reset_seeds, policy_seeds = _deployment_seed_vectors(task, config)
        pair_payload, pair_sha256 = pair_payloads[
            (task, selection.selected_opaque_id)
        ]
        private_record = registry.get(selection.selected_opaque_id)
        pair_returns = tuple(float(value) for value in pair_payload["episode_returns"])
        outcome = DeploymentResult(
            selection_id=selection.selection_id,
            protocol_id=selection.protocol_id,
            selected_opaque_id=selection.selected_opaque_id,
            status=str(pair_payload["status"]),
            deployment_failure=(
                None
                if pair_payload["deployment_failure"] is None
                else str(pair_payload["deployment_failure"])
            ),
            target_observation_dim=schemas[task].observation_dim,
            target_action_dim=schemas[task].action_dim,
            policy_observation_dim=private_record.native_observation_dim,
            policy_action_dim=private_record.native_action_dim,
            episode_returns=pair_returns,
            mean_return=(fmean(pair_returns) if pair_returns else None),
            return_std=(pstdev(pair_returns) if pair_returns else None),
            runtime_seconds=float(pair_payload["runtime_seconds"]),
            evaluation_reset_seeds=reset_seeds,
            evaluation_policy_seeds=policy_seeds,
        )
        outcome_digest = layout.publish_json(
            layout.deployment_result(query_id), outcome.to_dict(), resume=args.resume
        )
        outcomes.append(outcome)
        records.append(
            {
                "query_id": query_id,
                "target_task": task,
                "selected_opaque_id": selection.selected_opaque_id,
                "correct_retrieval": bool(raw["correct"]),
                "status": outcome.status,
                "deployment_failure": outcome.deployment_failure,
                "mean_return": outcome.mean_return,
                "deployment_sha256": outcome_digest,
                "pair_id": pair_payload["pair_id"],
                "pair_evaluation_sha256": pair_sha256,
            }
        )
    metrics = summarize_deployments(outcomes)
    correct_outcomes = [
        outcome
        for outcome, record in zip(outcomes, records, strict=True)
        if record["correct_retrieval"]
    ]
    correct_count = len(correct_outcomes)
    correct_deployable_count = sum(outcome.deployable for outcome in correct_outcomes)
    correct_deployability_rate = (
        correct_deployable_count / correct_count if correct_count else None
    )
    correct_summary = {
        "query_count": correct_count,
        "deployable_count": correct_deployable_count,
        "deployability_rate": correct_deployability_rate,
    }
    deployment_decision = deployment_gate(
        correct_retrieval_count=correct_count,
        correct_retrieval_deployability_rate=correct_deployability_rate,
        minimum_correct_retrieval_deployability_rate=(
            config.gates.deployment.minimum_correct_retrieval_deployability_rate
        ),
    )
    payload = {
        "schema": "policy-learnware.deployment-metrics.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": pool.pool_id,
        "public_pool_digest": sha256_json(pool.public_manifest()),
        "private_registry_sha256": sha256_file(layout.private_registry),
        "retrieval_metrics_sha256": sha256_file(layout.retrieval_metrics),
        "final_return_episode_count": config.episodes.final_return_per_task,
        "evaluation_mode": "unique-target-task-selected-policy-pair",
        "unique_pair_count": len(pair_plan),
        "pair_evaluations": {
            str(item["pair_id"]): sha256_file(
                layout.deployment_pair_evaluation(str(item["pair_id"]))
            )
            for item in pair_plan
        },
        "evaluator_contract": canonicalize(evaluator_contract),
        "selected_only": True,
        "fallback": "none",
        "summary": asdict(metrics),
        "correct_retrieval_deployment": correct_summary,
        "gate": deployment_decision.to_dict(),
        "gate_passed": deployment_decision.passed,
        "queries": records,
    }
    digest = layout.publish_json(layout.deployment_metrics, payload)
    _require_gate_passed(
        payload,
        expected_name="selected_policy_deployment",
        artifact=layout.deployment_metrics,
    )
    return {
        "metrics": str(layout.deployment_metrics),
        "sha256": digest,
        "deployability_rate": metrics.deployability_rate,
        "correct_retrieval_deployability_rate": correct_deployability_rate,
        "gate_passed": deployment_decision.passed,
        "query_count": metrics.query_count,
        "unique_pair_count": len(pair_plan),
        "evaluated_pair_count": evaluated_pair_count,
        "resumed_pair_count": resumed_pair_count,
    }


def _handle_build_report(
    args: argparse.Namespace, config: ProtocolDraft, layout: ArtifactLayout
) -> dict[str, Any]:
    protocol = _load_frozen_protocol(layout, config)
    selection_paths = [
        layout.selection_result(_query_id(task, bank, episode_count))
        for bank in range(config.episodes.target_query_banks)
        for task in config.environment.tasks
        for episode_count in config.episodes.target_query_prefix_grid
    ]
    retrieval = _load_verified_retrieval_artifacts(
        layout, config, selection_paths
    )
    deployment = _validate_deployment_resume(layout, config, retrieval)
    diagnostics = read_json(layout.unreduced_diagnostics)
    ranking = read_json(layout.reduced_unreduced_ranking)
    if not all(
        isinstance(value, Mapping)
        for value in (retrieval, deployment, diagnostics, ranking)
    ):
        raise CommandFailure("report inputs are not JSON objects")
    if (
        retrieval.get("schema") != "policy-learnware.retrieval-metrics.v0"
        or retrieval.get("complete") is not True
        or retrieval.get("protocol_draft_hash") != config.draft_hash
        or retrieval.get("protocol_id") != protocol.protocol_id
        or deployment.get("schema")
        != "policy-learnware.deployment-metrics.v0"
        or deployment.get("complete") is not True
        or deployment.get("protocol_draft_hash") != config.draft_hash
        or deployment.get("protocol_id") != protocol.protocol_id
        or deployment.get("retrieval_metrics_sha256")
        != sha256_file(layout.retrieval_metrics)
        or diagnostics.get("schema")
        != "policy-learnware.unreduced-diagnostics.v0"
        or diagnostics.get("complete") is not True
        or diagnostics.get("protocol_draft_hash") != config.draft_hash
        or diagnostics.get("protocol_id") != protocol.protocol_id
        or diagnostics.get("mmd_matrix_sha256") != sha256_file(layout.mmd_matrix)
        or ranking.get("schema")
        != "policy-learnware.reduced-unreduced-ranking.v0"
        or ranking.get("complete") is not True
        or ranking.get("protocol_draft_hash") != config.draft_hash
        or ranking.get("protocol_id") != protocol.protocol_id
        or ranking.get("task_rkme_sha256")
        != {
            task: sha256_file(layout.task_rkme(task))
            for task in config.environment.tasks
        }
    ):
        raise CommandFailure("report input provenance/digest binding mismatch")
    _validate_unreduced_diagnostics_artifact(
        diagnostics,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    _validate_ranking_artifact(
        ranking,
        config,
        layout,
        expected_protocol_id=protocol.protocol_id,
    )
    _require_gate_passed(
        retrieval,
        expected_name="exact_recurrent_retrieval",
        artifact=layout.retrieval_metrics,
    )
    _require_gate_passed(
        deployment,
        expected_name="selected_policy_deployment",
        artifact=layout.deployment_metrics,
    )
    retrieval_overall = retrieval["overall"]
    deployment_summary = deployment["summary"]
    lines = [
        "# Policy Learnware v0 实验报告",
        "",
        f"- Pool: `{config.pool.pool_id}`",
        f"- Protocol: `{protocol.protocol_id}`",
        f"- Exact-recurrent retrieval: {retrieval_overall['correct_count']}/"
        f"{retrieval_overall['trial_count']} "
        f"({100.0 * float(retrieval_overall['accuracy']):.2f}%)",
        f"- Deployability: {deployment_summary['deployable_count']}/"
        f"{deployment_summary['query_count']} "
        f"({100.0 * float(deployment_summary['deployability_rate']):.2f}%)",
        f"- Unreduced gate: {'PASS' if diagnostics['gate_passed'] else 'FAIL'}",
        f"- Reduced/unreduced ranking gate: "
        f"{'PASS' if ranking['gate_passed'] else 'FAIL'} "
        f"(top-1 agreement {100.0 * float(ranking['top1_agreement']):.2f}%)",
        f"- Max-prefix retrieval gate: "
        f"{'PASS' if retrieval['gate_passed'] else 'FAIL'}",
        f"- Correct-retrieval deployment gate: "
        f"{'PASS' if deployment['gate_passed'] else 'FAIL'}",
        f"- Minimum between-task MMD: {float(diagnostics['minimum_between_mmd']):.6g}",
        f"- Maximum within-task MMD: {float(diagnostics['maximum_within_mmd']):.6g}",
        "",
        "## Retrieval by probe episode budget",
        "",
        "| Episodes | Accuracy | Mean selected distance |",
        "|---:|---:|---:|",
    ]
    for count in config.episodes.target_query_prefix_grid:
        item = retrieval["by_prefix"][str(count)]
        lines.append(
            f"| {count} | {100.0 * float(item['accuracy']):.2f}% | "
            f"{float(item['mean_selected_distance']):.6g} |"
        )
    lines.extend(
        [
            "",
            "Selector只读取公开 TaskSpec RKME；算法、seed、return 与 policy 路径只存在于私有构建/部署制品。部署严格执行所选 policy，不回退第二候选。",
            "",
        ]
    )
    rendered = "\n".join(lines)
    if args.resume and layout.summary.is_file():
        if layout.summary.read_text(encoding="utf-8") != rendered:
            raise ArtifactLayoutError("report resume content differs from current inputs")
        return {"resumed": True, "summary": str(layout.summary)}
    _assert_output_state([layout.summary], resume=args.resume)
    digest = layout.publish_text(layout.summary, rendered)
    return {"summary": str(layout.summary), "sha256": digest}


HANDLERS = {
    "validate-config": _handle_validate_config,
    "smoke": _handle_smoke,
    "inspect-envs": _handle_inspect_envs,
    "collect-probe": _handle_collect_probe,
    "fit-normalizer": _handle_fit_normalizer,
    "train-encoder": _handle_train_encoder,
    "calibrate-kernel": _handle_calibrate_kernel,
    "diagnose-unreduced": _handle_diagnose_unreduced,
    "reduce-task-specs": _handle_reduce_task_specs,
    "inventory-policies": _handle_inventory_policies,
    "verify-policy-bundles": _handle_verify_policy_bundles,
    "championize": _handle_championize,
    "build-pool": _handle_build_pool,
    "evaluate-retrieval": _handle_evaluate_retrieval,
    "evaluate-deployment": _handle_evaluate_deployment,
    "build-report": _handle_build_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)
    try:
        config = load_protocol_draft(args.config)
        layout = ArtifactLayout(args.artifacts_root, config.pool.pool_id)
        _validate_command_arguments(args, config)
        if args.dry_run:
            _emit(_dry_run_payload(command, args, config, layout))
            return 0
        if command not in SUPPORTED_COMMANDS:
            raise CommandUnavailable(UNAVAILABLE_REASONS[command])
        result = HANDLERS[command](args, config, layout)
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "ok",
                "command": command,
                "protocol_draft_hash": config.draft_hash,
                "pool_id": config.pool.pool_id,
                "result": result,
            }
        )
        return 0
    except (ConfigError, CommandFailure, ArtifactLayoutError, ArtifactExistsError) as error:
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "error",
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error),
                "fail_closed": True,
            },
            stream=sys.stderr,
        )
        return 1
    except Exception as error:
        _emit(
            {
                "schema": CLI_SCHEMA,
                "status": "error",
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error),
                "fail_closed": True,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
