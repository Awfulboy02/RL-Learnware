#!/usr/bin/env python3
"""Collect the frozen v0.2 dynamics contexts with the CP0 Gaussian probe.

The runner intentionally publishes only one immutable bank and one one-line
index record per context.  Context directories are staged and renamed as a
unit, so independent shards can resume without a shared mutable ledger.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.config import ProtocolDraft, load_protocol_draft
from policy_learnware_v0.envs.mujoco_playground import (
    MujocoPlaygroundEnvAdapter,
    mujoco_playground_package_version,
)
from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v02.axes import DEVELOPMENT_ROLE, SOURCE_ROLE, AxisRegistry
from policy_learnware_v0.v02.axis_catalog import build_candidate_axis_catalog
from policy_learnware_v0.v02.axis_integration import axis_registry_from_config
from policy_learnware_v0.v02.config import (
    V02ExperimentConfig,
    load_v02_formal_config,
)
from policy_learnware_v0.v02.variant_env import VariantBuild, VariantEnvironmentFactory


INDEX_SCHEMA = "policy-learnware.v03-dynamics-probe-index.v0"
FAILURE_SCHEMA = "policy-learnware.v03-dynamics-probe-failure.v0"
CONTEXT_INDEX_SCHEMA = "policy-learnware.v03-dynamics-context-index.v0"
SEED_NAMESPACE = "v03-dynamics-probe-bank-v0"
SEED_BASE = 2_000_000_000
SEED_CONTEXT_STRIDE = 10_000
SEED_STREAM_OFFSET = 5_000
MAX_EPISODES = SEED_STREAM_OFFSET
TASK_TAXONOMY = {
    "CartpoleSwingup": ("cartpole", "swingup"),
    "CheetahRun": ("cheetah", "run"),
    "FingerTurnEasy": ("finger", "turn_easy"),
    "FishSwim": ("fish", "swim"),
    "ReacherEasy": ("reacher", "easy"),
    "WalkerWalk": ("walker", "walk"),
}


@dataclass(frozen=True)
class ProbeContext:
    context_id: str
    context_index: int
    role: str
    task_id: str
    axis_id: str
    factor_id: str
    factor_value: float
    source_anchor_id: str | None
    target_id: str | None

    def __post_init__(self) -> None:
        if self.role not in {SOURCE_ROLE, DEVELOPMENT_ROLE}:
            raise ValueError(f"unsupported probe context role: {self.role!r}")
        if self.role == SOURCE_ROLE:
            if self.source_anchor_id != self.context_id or self.target_id is not None:
                raise ValueError("source context identity is inconsistent")
        elif self.target_id != self.context_id or self.source_anchor_id is not None:
            raise ValueError("development context identity is inconsistent")

    def identity(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_index": self.context_index,
            "role": self.role,
            "task_id": self.task_id,
            "axis_id": self.axis_id,
            "factor_id": self.factor_id,
            "factor_value": self.factor_value,
            "source_anchor_id": self.source_anchor_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_index(cls, value: Mapping[str, Any]) -> "ProbeContext":
        return cls(
            context_id=str(value["context_id"]),
            context_index=int(value["context_index"]),
            role=str(value["role"]),
            task_id=str(value["task_id"]),
            axis_id=str(value["axis_id"]),
            factor_id=str(value["factor_id"]),
            factor_value=float(value["factor_value"]),
            source_anchor_id=(
                None
                if value.get("source_anchor_id") is None
                else str(value["source_anchor_id"])
            ),
            target_id=(
                None if value.get("target_id") is None else str(value["target_id"])
            ),
        )


@dataclass(frozen=True)
class CollectionPlan:
    v02_config: V02ExperimentConfig
    cp0_config: ProtocolDraft
    axis_registry: AxisRegistry
    contexts: tuple[ProbeContext, ...]


def _reviewed_catalog(config: V02ExperimentConfig) -> AxisRegistry:
    triples = sorted(
        {
            tuple(factor.value for factor in config.source_factors[task][axis.axis_id])
            for task in config.tasks
            for axis in config.dynamics_axes[task]
        }
    )
    if not triples:
        raise ValueError("frozen v0.2 config has no source factor grid")
    catalog, _ = build_candidate_axis_catalog(triples[0])
    return catalog


def build_probe_contexts(config: V02ExperimentConfig) -> tuple[ProbeContext, ...]:
    """Return the exact 30 source and 24 development contexts, globally sorted."""

    source_candidates: dict[str, list[ProbeContext]] = {}
    for task_id in config.tasks:
        for axis in config.dynamics_axes[task_id]:
            for factor in config.source_factors[task_id][axis.axis_id]:
                row = ProbeContext(
                    context_id=factor.source_anchor_id,
                    context_index=-1,
                    role=SOURCE_ROLE,
                    task_id=task_id,
                    axis_id=axis.axis_id,
                    factor_id=factor.factor_id,
                    factor_value=factor.value,
                    source_anchor_id=factor.source_anchor_id,
                    target_id=None,
                )
                source_candidates.setdefault(factor.source_anchor_id, []).append(row)

    sources: list[ProbeContext] = []
    for anchor_id, candidates in source_candidates.items():
        first = candidates[0]
        if any(
            item.task_id != first.task_id or item.factor_value != first.factor_value
            for item in candidates[1:]
        ):
            raise ValueError(f"source anchor {anchor_id} has inconsistent frozen rows")
        # The nominal anchor is deliberately shared by two axes.  Either axis
        # constructs the same native nominal instance; choose one stably.
        sources.append(
            min(candidates, key=lambda item: (item.axis_id, item.factor_id))
        )

    developments = [
        ProbeContext(
            context_id=target.target_id,
            context_index=-1,
            role=DEVELOPMENT_ROLE,
            task_id=target.task_id,
            axis_id=str(target.axis_id),
            factor_id=target.factor_id,
            factor_value=target.factor_value,
            source_anchor_id=None,
            target_id=target.target_id,
        )
        for target in config.development_targets
    ]
    if any(item.axis_id == "None" for item in developments):
        raise ValueError("development dynamics contexts must name an axis")
    if len(sources) != 30 or len(developments) != 24:
        raise ValueError(
            "frozen v0.2 collection scope must contain 30 source anchors and "
            f"24 development contexts, got {len(sources)} and {len(developments)}"
        )
    raw = sorted((*sources, *developments), key=lambda item: item.context_id)
    if len({item.context_id for item in raw}) != 54:
        raise ValueError("source and development context IDs must be disjoint")
    return tuple(
        replace(context, context_index=index) for index, context in enumerate(raw)
    )


def load_collection_plan(
    v02_config_path: str | Path,
    cp0_config_path: str | Path,
) -> CollectionPlan:
    v02_config = load_v02_formal_config(v02_config_path)
    cp0_config = load_protocol_draft(cp0_config_path)
    if cp0_config.environment.backend != MujocoPlaygroundEnvAdapter.backend:
        raise ValueError("CP0 config does not name the MuJoCo Playground backend")
    axis_registry = axis_registry_from_config(
        v02_config, _reviewed_catalog(v02_config)
    )
    return CollectionPlan(
        v02_config=v02_config,
        cp0_config=cp0_config,
        axis_registry=axis_registry,
        contexts=build_probe_contexts(v02_config),
    )


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if hasattr(value, "items"):
        return {str(key): _json_ready(item) for key, item in value.items()}
    raise TypeError(f"cannot convert {type(value).__name__} to canonical JSON")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_variant_factory(plan: CollectionPlan) -> VariantEnvironmentFactory:
    """Build the shared v0.2 native-variant factory for collection/evaluation."""

    from mujoco_playground import dm_control_suite, registry

    del dm_control_suite  # importing it registers the DMC suite
    defaults = {
        task: registry.get_default_config(task) for task in plan.v02_config.tasks
    }
    registry_config_digests = {
        task: sha256_json(_json_ready(config)) for task, config in defaults.items()
    }
    runtime = {
        "fpo_commit": plan.cp0_config.runtime.fpo_commit,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "mujoco": _package_version("mujoco"),
        "playground": mujoco_playground_package_version(),
    }

    def nominal_loader(task: str) -> Any:
        return registry.load(task, config=registry.get_default_config(task))

    def adapter_factory(native: Any, task: str, *, jit: bool) -> Any:
        return MujocoPlaygroundEnvAdapter(
            task,
            native_environment=native,
            expected_horizon=plan.cp0_config.environment.horizon,
            expected_action_repeat=plan.cp0_config.environment.action_repeat,
            jit=jit,
        )

    return VariantEnvironmentFactory(
        registry=plan.axis_registry,
        nominal_loader=nominal_loader,
        adapter_factory=adapter_factory,
        registry_config_digests=registry_config_digests,
        runtime_digest=sha256_json(runtime),
    )


def build_environment(
    plan: CollectionPlan,
    context: ProbeContext | Mapping[str, Any],
    *,
    factory: VariantEnvironmentFactory | None = None,
    jit: bool = False,
) -> VariantBuild:
    """Construct one reviewed context; shared by collector and baseline rollout."""

    resolved = (
        context if isinstance(context, ProbeContext) else ProbeContext.from_index(context)
    )
    build = (factory or build_variant_factory(plan)).create(
        task_id=resolved.task_id,
        axis_id=resolved.axis_id,
        factor_id=resolved.factor_id,
        role=resolved.role,  # type: ignore[arg-type]
        jit=jit,
    )
    if build.factor_value != resolved.factor_value:
        raise ValueError(f"factor value changed for context {resolved.context_id}")
    if resolved.role == SOURCE_ROLE:
        if build.source_anchor_id != resolved.source_anchor_id:
            raise ValueError(
                f"live source identity changed for context {resolved.context_id}"
            )
    elif build.source_anchor_id is not None:
        raise ValueError("development build unexpectedly produced a source anchor ID")
    return build


def _episode_seeds(
    context: ProbeContext, episode_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= episode_count <= MAX_EPISODES:
        raise ValueError(f"episodes must lie in [1, {MAX_EPISODES}]")
    start = SEED_BASE + context.context_index * SEED_CONTEXT_STRIDE
    reset = start + np.arange(episode_count, dtype=np.int64)
    probe = start + SEED_STREAM_OFFSET + np.arange(episode_count, dtype=np.int64)
    return reset, probe


def _static_index(
    plan: CollectionPlan,
    context: ProbeContext,
    *,
    episode_count: int,
    steps: int,
    sigma: float,
) -> dict[str, Any]:
    return {
        "schema": INDEX_SCHEMA,
        "status": "COMPLETE",
        **context.identity(),
        "v02_config_digest": plan.v02_config.config_digest,
        "axis_registry_digest": plan.axis_registry.digest,
        "frozen_probe_protocol_id": plan.v02_config.probe_protocol_id,
        "cp0_draft_hash": plan.cp0_config.draft_hash,
        "probe_type": plan.cp0_config.probe.type,
        "probe_rng_backend": plan.cp0_config.probe.rng_backend,
        "probe_sigma": sigma,
        "seed_namespace": SEED_NAMESPACE,
        "seed_context_index": context.context_index,
        "episode_count": episode_count,
        "steps_per_episode": steps,
    }


def _resume_context(
    destination: Path,
    expected: Mapping[str, Any],
    context: ProbeContext,
) -> None:
    index_path = destination / "index.json"
    bank_path = destination / "bank.npz"
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"resume context is not a regular directory: {destination}")
    for path in (index_path, bank_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"resume context lacks a regular {path.name}: {destination}")
    row = read_json(index_path)
    if not isinstance(row, Mapping):
        raise ValueError(f"index row is not an object: {index_path}")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"resume index mismatch at {context.context_id}.{key}")
    if row.get("npz_path") != "bank.npz":
        raise ValueError(f"resume NPZ path changed for {context.context_id}")
    actual_npz_sha = sha256_file(bank_path)
    if row.get("bank_npz_sha256") != actual_npz_sha:
        raise ValueError(f"resume NPZ digest mismatch for {context.context_id}")
    dataset = EpisodeDataset.load_npz(bank_path)
    reset, probe = _episode_seeds(context, int(expected["episode_count"]))
    checks = {
        "dataset_digest": dataset.digest,
        "transition_count": dataset.transition_count,
        "observation_dim": dataset.observation_dim,
        "action_dim": dataset.action_dim,
    }
    for key, value in checks.items():
        if row.get(key) != value:
            raise ValueError(f"resume dataset mismatch at {context.context_id}.{key}")
    if not np.array_equal(dataset.reset_seeds, reset) or not np.array_equal(
        dataset.probe_seeds, probe
    ):
        raise ValueError(f"resume seed mismatch for {context.context_id}")


def _publish_context(
    destination: Path,
    *,
    row: Mapping[str, Any],
    dataset: EpisodeDataset,
) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        npz_sha = dataset.save_npz(staging / "bank.npz", overwrite=False)
        final = {
            **row,
            "npz_path": "bank.npz",
            "bank_npz_sha256": npz_sha,
            "dataset_digest": dataset.digest,
            "transition_count": dataset.transition_count,
            "observation_dim": dataset.observation_dim,
            "action_dim": dataset.action_dim,
        }
        atomic_write_json(staging / "index.json", final, overwrite=False)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _expected_source(plan: CollectionPlan, context: ProbeContext) -> str | None:
    if context.role == SOURCE_ROLE:
        return None
    candidates = plan.v02_config.source_factors[context.task_id][context.axis_id]
    selected = min(
        candidates,
        key=lambda item: (
            abs(item.value - context.factor_value),
            item.value,
            item.source_anchor_id,
        ),
    )
    return selected.source_anchor_id


def finalize_context_index(
    output_dir: str | Path,
    plan: CollectionPlan,
    *,
    resume: bool = False,
) -> Path:
    """Merge 54 atomic context rows after all collection shards complete."""

    root = Path(output_dir).expanduser().resolve(strict=True)
    contexts_root = root / "contexts"
    if contexts_root.is_symlink() or not contexts_root.is_dir():
        raise ValueError("collector contexts directory is absent or unsafe")
    paths = sorted(contexts_root.glob("*/index.json"))
    if len(paths) != len(plan.contexts):
        raise ValueError(
            f"context index requires {len(plan.contexts)} complete rows, found {len(paths)}"
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe context index row: {path}")
        row = read_json(path)
        if not isinstance(row, Mapping) or row.get("status") != "COMPLETE":
            raise ValueError(f"incomplete context index row: {path}")
        context_id = str(row.get("context_id", ""))
        if context_id in rows:
            raise ValueError(f"duplicate context row: {context_id}")
        rows[context_id] = row

    merged: list[dict[str, Any]] = []
    for context in plan.contexts:
        row = rows.get(context.context_id)
        if row is None:
            raise ValueError(f"missing context row: {context.context_id}")
        required = {
            **context.identity(),
            "v02_config_digest": plan.v02_config.config_digest,
            "axis_registry_digest": plan.axis_registry.digest,
            "cp0_draft_hash": plan.cp0_config.draft_hash,
            "npz_path": "bank.npz",
        }
        for key, value in required.items():
            if row.get(key) != value:
                raise ValueError(f"context row mismatch at {context.context_id}.{key}")
        bank_path = contexts_root / context.context_id / "bank.npz"
        if bank_path.is_symlink() or not bank_path.is_file():
            raise ValueError(f"context bank is absent or unsafe: {bank_path}")
        try:
            embodiment, goal = TASK_TAXONOMY[context.task_id]
        except KeyError as error:
            raise ValueError(f"task taxonomy is missing {context.task_id}") from error
        signal_row = {
            "bank_id": context.context_id,
            "context_id": context.context_id,
            "role": (
                "source" if context.role == SOURCE_ROLE else "development_query"
            ),
            "task_id": context.task_id,
            "embodiment_id": embodiment,
            "goal_id": goal,
            "dynamics_id": (
                "nominal"
                if context.factor_value == 1.0
                else f"{context.axis_id}:{context.factor_id}"
            ),
            "axis_id": context.axis_id,
            "factor_id": context.factor_id,
            "factor_value": context.factor_value,
            "npz": str(bank_path.relative_to(root)),
            "dataset_digest": row["dataset_digest"],
            "bank_npz_sha256": row["bank_npz_sha256"],
        }
        expected_source = _expected_source(plan, context)
        if expected_source is not None:
            signal_row["expected_source_bank_id"] = expected_source
        merged.append(signal_row)
    if set(rows) != {item.context_id for item in plan.contexts}:
        raise ValueError("context directory contains rows outside the frozen 54 contexts")
    payload = {
        "schema": CONTEXT_INDEX_SCHEMA,
        "v02_config_digest": plan.v02_config.config_digest,
        "cp0_draft_hash": plan.cp0_config.draft_hash,
        "contexts": merged,
    }
    destination = root / "context_index.json"
    expected_bytes = canonical_json_bytes(payload) + b"\n"
    if destination.exists():
        if not resume:
            raise ValueError("context_index.json exists; pass --resume to verify it")
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != expected_bytes
        ):
            raise ValueError("existing context_index.json differs from complete contexts")
    else:
        atomic_write_json(destination, payload, overwrite=False)
    return destination


def _record_failure(
    failures_root: Path,
    *,
    context: ProbeContext,
    episode_count: int,
    steps: int,
    sigma: float,
    error: Exception,
) -> Path:
    payload = {
        "schema": FAILURE_SCHEMA,
        "status": "FAILED",
        **context.identity(),
        "episode_count": episode_count,
        "steps_per_episode": steps,
        "probe_sigma": sigma,
        "seed_namespace": SEED_NAMESPACE,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    failure_id = sha256_json(payload)[:16]
    path = failures_root / f"{context.context_id}-{failure_id}.json"
    expected = canonical_json_bytes(payload) + b"\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"existing failure evidence differs: {path}")
    else:
        atomic_write_json(path, payload, overwrite=False)
    return path


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v02-config", type=Path, default=Path("configs/v02_freeze_ready.yaml")
    )
    parser.add_argument(
        "--cp0-config", type=Path, default=Path("configs/dmc6_outer006_v0.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=_positive_int, default=32)
    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument(
        "--sigma",
        type=_positive_float,
        default=None,
        help="Gaussian sigma; default is the frozen CP0 value (1.0)",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=_positive_int, default=1)
    parser.add_argument("--max-contexts", type=_positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--finalize-index",
        action="store_true",
        help="merge 54 completed atomic rows without loading an environment",
    )
    return parser


def _prepare_output(path: Path) -> tuple[Path, Path]:
    root = path.expanduser().resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("output-dir must be a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    contexts = root / "contexts"
    failures = root / "failures"
    contexts.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    if contexts.is_symlink() or failures.is_symlink():
        raise ValueError("collector output subdirectories cannot be symlinks")
    return contexts, failures


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    if args.episodes > MAX_EPISODES:
        raise ValueError(f"episodes cannot exceed {MAX_EPISODES}")

    plan = load_collection_plan(args.v02_config, args.cp0_config)
    if args.finalize_index:
        destination = finalize_context_index(
            args.output_dir, plan, resume=args.resume
        )
        print(json.dumps({"status": "COMPLETE", "context_index": str(destination)}))
        return 0
    if args.steps > plan.cp0_config.environment.horizon:
        raise ValueError("steps cannot exceed the frozen environment horizon")
    sigma = (
        plan.cp0_config.probe.sigma if args.sigma is None else float(args.sigma)
    )
    scheduled = tuple(
        context
        for context in plan.contexts
        if context.context_index % args.shard_count == args.shard_index
    )
    if args.max_contexts is not None:
        scheduled = scheduled[: args.max_contexts]
    contexts_root, failures_root = _prepare_output(args.output_dir)
    if not args.resume:
        existing = [
            context.context_id
            for context in scheduled
            if (contexts_root / context.context_id).exists()
        ]
        if existing:
            raise ValueError(
                "scheduled outputs already exist; pass --resume: " + ", ".join(existing)
            )

    factory: VariantEnvironmentFactory | None = None
    completed = resumed = failed = 0
    for context in scheduled:
        destination = contexts_root / context.context_id
        static = _static_index(
            plan,
            context,
            episode_count=args.episodes,
            steps=args.steps,
            sigma=sigma,
        )
        try:
            if destination.exists():
                if not args.resume:
                    raise ValueError(f"context output already exists: {destination}")
                _resume_context(destination, static, context)
                resumed += 1
                print(json.dumps({"status": "RESUMED", "context_id": context.context_id}))
                continue

            if factory is None:
                factory = build_variant_factory(plan)
            build = build_environment(plan, context, factory=factory, jit=False)
            schema = build.adapter.schema
            if not np.allclose(schema.action_low, plan.cp0_config.probe.action_low) or not np.allclose(
                schema.action_high, plan.cp0_config.probe.action_high
            ):
                raise ValueError("native action bounds differ from the frozen CP0 probe")
            reset_seeds, probe_seeds = _episode_seeds(context, args.episodes)
            arrays = build.adapter.collect_clipped_gaussian_batch(
                reset_seeds=reset_seeds,
                probe_seeds=probe_seeds,
                sigma=sigma,
                steps=args.steps,
            )
            dataset = EpisodeDataset(
                **arrays,
                episode_offsets=np.arange(
                    0, (args.episodes + 1) * args.steps, args.steps, dtype=np.int64
                ),
                reset_seeds=reset_seeds,
                probe_seeds=probe_seeds,
            )
            row = {
                **static,
                "environment_instance_digest": build.environment_instance_digest,
            }
            _publish_context(destination, row=row, dataset=dataset)
            completed += 1
            print(json.dumps({"status": "COMPLETE", "context_id": context.context_id}))
        except Exception as error:  # keep other immutable context work running
            failed += 1
            evidence = _record_failure(
                failures_root,
                context=context,
                episode_count=args.episodes,
                steps=args.steps,
                sigma=sigma,
                error=error,
            )
            print(
                json.dumps(
                    {
                        "status": "FAILED",
                        "context_id": context.context_id,
                        "error": str(error),
                        "evidence": str(evidence),
                    }
                ),
                file=sys.stderr,
            )

    print(
        json.dumps(
            {
                "status": "FAILED" if failed else "COMPLETE",
                "scheduled": len(scheduled),
                "completed": completed,
                "resumed": resumed,
                "failed": failed,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CollectionPlan",
    "ProbeContext",
    "build_environment",
    "build_probe_contexts",
    "build_variant_factory",
    "finalize_context_index",
    "load_collection_plan",
]
