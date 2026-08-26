"""Minimal resumable runner for the frozen v0.3 45-fit representation plan.

The runner deliberately stops at trained representation checkpoints.  It
loads the already-frozen legacy v0 encoder train/validation datasets, applies
the v0.3 global canonicalizer and condition transforms, and executes the real
JAX R5/R5L trainers.  Each completed fit is published as one immutable job
directory; a small per-shard summary is the only mutable progress artifact.

No formal-review authority is minted here.  The resulting checkpoints are
inputs to the separately authorized Signal Atlas run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import (
    sha256_file,
    sha256_json,
)
from policy_learnware_v0.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from policy_learnware_v0.probe.dataset import load_dataset_artifact
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.corro_trainers import (
    TASK_SUPCON_OBJECTIVE_DIGEST,
    CorroOptimizationConfig,
    CorroSourceSplit,
    CorroTaskDataset,
    CorroTrainerAdapter,
)
from policy_learnware_v0.v03.representation_ladder import (
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationBatch,
    RepresentationManifest,
    TrainingRequest,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
)
from policy_learnware_v0.v03.signal_controls import (
    HistoricalRandomTanhSpec,
    RewardFreeShuffledNextSpec,
)
from policy_learnware_v0.v03.signal_matrix import (
    C_RF_SHUFFLED_NEXT,
    SignalFitJob,
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    TransitionBank,
    apply_transition_view,
)


JOB_RECORD_SCHEMA = "policy-learnware.v03-server-signal-fit-job.v0"
SUMMARY_SCHEMA = "policy-learnware.v03-server-signal-fit-summary.v0"
SOURCE_ROLE_BY_SPLIT = {
    "encoder_train": "source_representation_train",
    "encoder_validation": "source_representation_validation",
}
SHARED_OUTPUT_DIM = 32
HIDDEN_DIMS = (256, 256)


class SignalFitRunnerError(RuntimeError):
    """The legacy input, persisted job, or requested execution is invalid."""


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer"
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def _episode_coordinates(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    episode_id = np.empty(int(offsets[-1]), dtype=np.int64)
    timestep = np.empty_like(episode_id)
    for index, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        start_i, stop_i = int(start), int(stop)
        episode_id[start_i:stop_i] = index
        timestep[start_i:stop_i] = np.arange(stop_i - start_i, dtype=np.int64)
    return episode_id, timestep


def _load_legacy_split(root: Path, split: str) -> tuple[NativeTransitionBank, ...]:
    split_root = root / "datasets" / split
    if not split_root.is_dir() or split_root.is_symlink():
        raise SignalFitRunnerError(f"missing regular legacy dataset split: {split_root}")
    manifests = sorted(split_root.glob("*.json"))
    if not manifests:
        raise SignalFitRunnerError(f"legacy split contains no manifests: {split_root}")
    role = SOURCE_ROLE_BY_SPLIT[split]
    banks: list[NativeTransitionBank] = []
    seen_tasks: set[str] = set()
    for manifest_path in manifests:
        npz_path = manifest_path.with_suffix(".npz")
        if (
            manifest_path.is_symlink()
            or npz_path.is_symlink()
            or not manifest_path.is_file()
            or not npz_path.is_file()
        ):
            raise SignalFitRunnerError(
                f"legacy dataset pair must be regular files: {manifest_path}"
            )
        try:
            dataset, manifest = load_dataset_artifact(npz_path, manifest_path)
        except (OSError, TypeError, ValueError) as error:
            raise SignalFitRunnerError(
                f"invalid legacy dataset {manifest_path}: {error}"
            ) from error
        if manifest.split != split or manifest_path.stem != manifest.task:
            raise SignalFitRunnerError(
                f"legacy dataset identity differs from its path: {manifest_path}"
            )
        if manifest.task in seen_tasks:
            raise SignalFitRunnerError(
                f"duplicate task in legacy {split}: {manifest.task}"
            )
        seen_tasks.add(manifest.task)
        episode_id, timestep = _episode_coordinates(dataset.episode_offsets)
        native_schema_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-legacy-native-abi.v0",
                "task_private_id": manifest.task,
                "observation_dim": dataset.observation_dim,
                "action_dim": dataset.action_dim,
            }
        )
        banks.append(
            NativeTransitionBank(
                bank_id=f"{split}-{manifest.task}",
                task_private_id=manifest.task,
                data_role=role,  # type: ignore[arg-type]
                native_schema_digest=native_schema_digest,
                raw_dataset_digest=dataset.digest,
                observation=dataset.observation,
                action=dataset.action,
                reward=dataset.reward,
                next_observation=dataset.next_observation,
                terminated=dataset.terminated,
                truncated=dataset.truncated,
                episode_id=episode_id,
                timestep=timestep,
            )
        )
    return tuple(sorted(banks, key=lambda item: item.task_private_id))


def _load_source_banks(
    legacy_root: Path,
) -> tuple[
    tuple[NativeTransitionBank, ...],
    tuple[NativeTransitionBank, ...],
    GlobalCanonicalizerSpec,
    tuple[TransitionBank, ...],
    tuple[TransitionBank, ...],
]:
    train_native = _load_legacy_split(legacy_root, "encoder_train")
    validation_native = _load_legacy_split(legacy_root, "encoder_validation")
    train_tasks = tuple(item.task_private_id for item in train_native)
    validation_tasks = tuple(item.task_private_id for item in validation_native)
    if train_tasks != validation_tasks or len(train_tasks) != 6:
        raise SignalFitRunnerError(
            "legacy encoder train/validation must contain the same exact six tasks"
        )
    for train, validation in zip(train_native, validation_native, strict=True):
        if train.raw_dataset_digest == validation.raw_dataset_digest:
            raise SignalFitRunnerError(
                f"{train.task_private_id}: train and validation datasets are identical"
            )
    all_source = (*train_native, *validation_native)
    registry = NativeShapeRegistry.from_source_banks(all_source)
    normalizer = fit_global_normalizer(all_source, registry=registry)
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    train = tuple(
        TransitionBank.from_canonical_batch(canonicalizer.transform(bank).batch)
        for bank in train_native
    )
    validation = tuple(
        TransitionBank.from_canonical_batch(canonicalizer.transform(bank).batch)
        for bank in validation_native
    )
    return train_native, validation_native, canonicalizer, train, validation


def _condition_plan(first_bank: TransitionBank) -> ConditionExecutionPlan:
    full = apply_transition_view(first_bank, V_FULL_LEGACY)
    historical = HistoricalRandomTanhSpec.create(
        seed=0,
        input_dim=int(full.feature_matrix.shape[1]),
        output_dim=SHARED_OUTPUT_DIM,
    )
    return ConditionExecutionPlan.create(historical_spec=historical)


def _condition_matrix(
    bank: TransitionBank,
    condition_id: str,
    condition_plan: ConditionExecutionPlan,
) -> np.ndarray:
    if condition_id == C_RF_SHUFFLED_NEXT:
        return RewardFreeShuffledNextSpec(
            seed=condition_plan.rf_shuffled_next_seed
        ).apply(bank).feature_matrix
    seed = condition_plan.transition_view_seeds.get(condition_id, 0)
    return apply_transition_view(
        bank,
        condition_id,
        shuffle_seed=int(seed),
    ).feature_matrix


def _source_split(
    *,
    role: str,
    native_banks: Sequence[NativeTransitionBank],
    transition_banks: Sequence[TransitionBank],
    condition_id: str,
    condition_plan: ConditionExecutionPlan,
) -> CorroSourceSplit:
    if len(native_banks) != len(transition_banks):
        raise SignalFitRunnerError("native/canonical source bank coverage differs")
    tasks = tuple(
        CorroTaskDataset(
            task_id=native.task_private_id,
            packed=_condition_matrix(bank, condition_id, condition_plan),
            episode_offsets=bank.episode_offsets,
        )
        for native, bank in zip(native_banks, transition_banks, strict=True)
    )
    return CorroSourceSplit(role=role, tasks=tasks)  # type: ignore[arg-type]


def _request(job: SignalFitJob, input_dim: int) -> TrainingRequest:
    if job.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS
        activation = "relu"
    elif job.representation_id == R5L_SUPERVISED_LINEAR:
        hidden_dims = ()
        activation = None
    else:  # build_optimization_fit_jobs owns this invariant
        raise SignalFitRunnerError(
            f"unsupported optimization representation: {job.representation_id}"
        )
    return TrainingRequest(
        representation_id=job.representation_id,
        input_dim=input_dim,
        output_dim=SHARED_OUTPUT_DIM,
        hidden_dims=hidden_dims,
        activation=activation,
        l2_normalize_output=True,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=job.seed,
    )


def _job_directory(output_dir: Path, job: SignalFitJob) -> Path:
    return output_dir / "jobs" / f"job-{job.job_digest}"


def _expected_resume_binding(
    *,
    plan_digest: str,
    job: SignalFitJob,
    optimization: CorroOptimizationConfig,
    trainer: CorroTrainerAdapter,
    source: RepresentationBatch,
    request: TrainingRequest,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "job_digest": job.job_digest,
        "plan_digest": plan_digest,
        "condition_id": job.condition_id,
        "representation_id": job.representation_id,
        "seed": job.seed,
        "train_steps": optimization.train_steps,
        "optimization_digest": optimization.optimization_digest,
        "trainer_adapter_digest": trainer.adapter_digest,
        "train_split_digest": trainer.train_split.split_digest,
        "validation_split_digest": trainer.validation_split.split_digest,
        "source_batch_digest": source.batch_digest,
        "training_request_digest": request.request_digest,
    }


def _verify_completed_job(
    destination: Path, expected: Mapping[str, Any]
) -> Mapping[str, Any]:
    record_path = destination / "job_record.json"
    checkpoint_path = destination / "checkpoint.bin"
    manifest_path = destination / "representation_manifest.json"
    for path in (destination, record_path, checkpoint_path, manifest_path):
        if path.is_symlink() or not path.exists():
            raise SignalFitRunnerError(
                f"resume target is missing or symlinked: {path}"
            )
    if not destination.is_dir() or not all(
        path.is_file() for path in (record_path, checkpoint_path, manifest_path)
    ):
        raise SignalFitRunnerError(f"resume target is not a complete job: {destination}")
    try:
        record = read_json(record_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SignalFitRunnerError(f"cannot read job record: {destination}") from error
    if not isinstance(record, Mapping) or record.get("schema") != JOB_RECORD_SCHEMA:
        raise SignalFitRunnerError(f"unsupported job record: {destination}")
    supplied_record_digest = record.get("record_digest")
    body = dict(record)
    body.pop("record_digest", None)
    if supplied_record_digest != sha256_json(body):
        raise SignalFitRunnerError(f"job record digest mismatch: {destination}")
    for name, value in expected.items():
        if record.get(name) != value:
            raise SignalFitRunnerError(
                f"resume job binding drifted at {name}: {destination}"
            )
    if record.get("status") != "COMPLETE":
        raise SignalFitRunnerError(f"resume job is not COMPLETE: {destination}")
    if record.get("checkpoint_file") != checkpoint_path.name or record.get(
        "representation_manifest_file"
    ) != manifest_path.name:
        raise SignalFitRunnerError(f"resume artifact names drifted: {destination}")
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    if checkpoint_sha != record.get("checkpoint_file_sha256") or manifest_sha != record.get(
        "representation_manifest_file_sha256"
    ):
        raise SignalFitRunnerError(f"resume artifact bytes drifted: {destination}")
    try:
        manifest = RepresentationManifest.from_dict(read_json(manifest_path))
    except (OSError, TypeError, ValueError) as error:
        raise SignalFitRunnerError(
            f"resume representation manifest is invalid: {destination}"
        ) from error
    if (
        manifest.coordinate_digest != record.get("representation_coordinate_digest")
        or manifest.checkpoint_digest != checkpoint_sha
        or manifest.protocol_digest != expected["training_request_digest"]
        or manifest.source_fit_digest != record.get("representation_source_fit_digest")
    ):
        raise SignalFitRunnerError(
            f"resume representation/checkpoint binding drifted: {destination}"
        )
    return record


def _publish_completed_job(
    *,
    output_dir: Path,
    plan_digest: str,
    job: SignalFitJob,
    optimization: CorroOptimizationConfig,
    trainer: CorroTrainerAdapter,
    source: RepresentationBatch,
    request: TrainingRequest,
    fitted: Any,
) -> Mapping[str, Any]:
    checkpoint_bytes = fitted.checkpoint_bytes
    if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
        raise SignalFitRunnerError(f"{job.job_id}: trainer returned no checkpoint bytes")
    destination = _job_directory(output_dir, job)
    jobs_root = destination.parent
    jobs_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=jobs_root))
    try:
        checkpoint_path = staging / "checkpoint.bin"
        manifest_path = staging / "representation_manifest.json"
        record_path = staging / "job_record.json"
        checkpoint_sha = atomic_write_bytes(checkpoint_path, checkpoint_bytes)
        manifest_sha = atomic_write_json(manifest_path, fitted.manifest.to_dict())
        body = {
            "schema": JOB_RECORD_SCHEMA,
            **_expected_resume_binding(
                plan_digest=plan_digest,
                job=job,
                optimization=optimization,
                trainer=trainer,
                source=source,
                request=request,
            ),
            "status": "COMPLETE",
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_file_sha256": checkpoint_sha,
            "checkpoint_digest": fitted.manifest.checkpoint_digest,
            "representation_manifest_file": manifest_path.name,
            "representation_manifest_file_sha256": manifest_sha,
            "representation_coordinate_digest": fitted.manifest.coordinate_digest,
            "representation_source_fit_digest": fitted.manifest.source_fit_digest,
        }
        record = {**body, "record_digest": sha256_json(body)}
        atomic_write_json(record_path, record)
        try:
            os.rename(staging, destination)
        except FileExistsError as error:
            raise SignalFitRunnerError(
                f"completed job already exists; use --resume: {destination}"
            ) from error
        return record
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _fit_one(
    *,
    output_dir: Path,
    plan_digest: str,
    job: SignalFitJob,
    optimization: CorroOptimizationConfig,
    train_native: Sequence[NativeTransitionBank],
    validation_native: Sequence[NativeTransitionBank],
    train_banks: Sequence[TransitionBank],
    validation_banks: Sequence[TransitionBank],
    condition_plan: ConditionExecutionPlan,
    resume: bool,
) -> tuple[str, Mapping[str, Any]]:
    train_split = _source_split(
        role="source_representation_train",
        native_banks=train_native,
        transition_banks=train_banks,
        condition_id=job.condition_id,
        condition_plan=condition_plan,
    )
    validation_split = _source_split(
        role="source_representation_validation",
        native_banks=validation_native,
        transition_banks=validation_banks,
        condition_id=job.condition_id,
        condition_plan=condition_plan,
    )
    trainer = CorroTrainerAdapter(
        train_split=train_split,
        validation_split=validation_split,
        optimization=optimization,
    )
    source = RepresentationBatch(
        train_split.flattened_values(),
        train_split.split_digest,
        "SOURCE_FIT",
    )
    request = _request(job, source.input_dim)
    expected = _expected_resume_binding(
        plan_digest=plan_digest,
        job=job,
        optimization=optimization,
        trainer=trainer,
        source=source,
        request=request,
    )
    destination = _job_directory(output_dir, job)
    if destination.exists():
        if not resume:
            raise SignalFitRunnerError(
                f"completed job already exists; use --resume: {destination}"
            )
        return "RESUMED", _verify_completed_job(destination, expected)
    labels = train_split.flattened_task_names()
    if job.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
        fitted = fit_r5_corro_style(
            source,
            labels=labels,
            trainer=trainer,
            objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
            seed=job.seed,
            output_dim=SHARED_OUTPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )
    else:
        fitted = fit_r5l_supervised_linear(
            source,
            labels=labels,
            trainer=trainer,
            objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
            seed=job.seed,
            output_dim=SHARED_OUTPUT_DIM,
        )
    if fitted.manifest.protocol_digest != request.request_digest:
        raise SignalFitRunnerError(f"{job.job_id}: fitted request binding drifted")
    record = _publish_completed_job(
        output_dir=output_dir,
        plan_digest=plan_digest,
        job=job,
        optimization=optimization,
        trainer=trainer,
        source=source,
        request=request,
        fitted=fitted,
    )
    return "TRAINED", record


def _write_summary(path: Path, body: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(body)
    payload["summary_digest"] = sha256_json(payload)
    atomic_write_json(path, payload, overwrite=True)
    return payload


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    legacy_root = Path(args.legacy_v0_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not legacy_root.is_dir() or legacy_root.is_symlink():
        raise SignalFitRunnerError(
            f"legacy-v0-root must be a regular directory: {legacy_root}"
        )
    if args.shard_index >= args.shard_count:
        raise SignalFitRunnerError("shard-index must be smaller than shard-count")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise SignalFitRunnerError("output-dir may not be a symlink")

    (
        train_native,
        validation_native,
        canonicalizer,
        train_banks,
        validation_banks,
    ) = _load_source_banks(legacy_root)
    condition_plan = _condition_plan(train_banks[0])
    signal_plan = build_signal_matrix_plan()
    all_jobs = build_optimization_fit_jobs(signal_plan)
    shard_jobs = tuple(
        job
        for index, job in enumerate(all_jobs)
        if index % args.shard_count == args.shard_index
    )
    scheduled_jobs = (
        shard_jobs
        if args.max_jobs is None
        else shard_jobs[: int(args.max_jobs)]
    )
    optimization = CorroOptimizationConfig(train_steps=args.train_steps)
    input_binding_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-server-fit-inputs.v0",
            "train_native_bank_digests": [
                item.native_bank_digest for item in train_native
            ],
            "validation_native_bank_digests": [
                item.native_bank_digest for item in validation_native
            ],
            "canonicalizer_digest": canonicalizer.canonicalizer_digest,
            "condition_plan_digest": condition_plan.plan_digest,
        }
    )
    summary_path = output_dir / (
        f"summary-shard-{args.shard_index:05d}-of-{args.shard_count:05d}.json"
    )
    completed: list[str] = []
    trained: list[str] = []
    resumed: list[str] = []
    failed: dict[str, str] = {}
    base_summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "status": "RUNNING",
        "signal_matrix_plan_digest": signal_plan.plan_digest,
        "logical_cell_count": signal_plan.logical_cell_count,
        "numeric_cell_count": signal_plan.numeric_cell_count,
        "structural_na_count": signal_plan.structural_na_count,
        "total_fit_job_count": len(all_jobs),
        "r5_fit_job_count": sum(
            item.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
            for item in all_jobs
        ),
        "r5l_fit_job_count": sum(
            item.representation_id == R5L_SUPERVISED_LINEAR for item in all_jobs
        ),
        "train_steps": optimization.train_steps,
        "optimization_digest": optimization.optimization_digest,
        "input_binding_digest": input_binding_digest,
        "task_private_ids": [item.task_private_id for item in train_native],
        "native_shape_registry_digest": canonicalizer.registry.registry_digest,
        "normalizer_digest": canonicalizer.normalizer.normalizer_digest,
        "canonicalizer_digest": canonicalizer.canonicalizer_digest,
        "condition_execution_plan_digest": condition_plan.plan_digest,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_job_count": len(shard_jobs),
        "scheduled_job_count": len(scheduled_jobs),
        "max_jobs": args.max_jobs,
        "completed_job_ids": completed,
        "trained_job_ids": trained,
        "resumed_job_ids": resumed,
        "failed_jobs": failed,
        "pending_job_ids": [item.job_id for item in scheduled_jobs],
        "deferred_job_ids": [
            item.job_id for item in all_jobs if item not in set(scheduled_jobs)
        ],
    }
    _write_summary(summary_path, base_summary)
    for job in scheduled_jobs:
        try:
            disposition, _record = _fit_one(
                output_dir=output_dir,
                plan_digest=str(signal_plan.plan_digest),
                job=job,
                optimization=optimization,
                train_native=train_native,
                validation_native=validation_native,
                train_banks=train_banks,
                validation_banks=validation_banks,
                condition_plan=condition_plan,
                resume=bool(args.resume),
            )
        except Exception as error:
            failed[job.job_id] = f"{type(error).__name__}: {error}"
            base_summary.update(
                {
                    "status": "FAILED",
                    "completed_job_ids": list(completed),
                    "trained_job_ids": list(trained),
                    "resumed_job_ids": list(resumed),
                    "failed_jobs": dict(failed),
                    "pending_job_ids": [
                        item.job_id
                        for item in scheduled_jobs
                        if item.job_id not in set(completed)
                        and item.job_id not in failed
                    ],
                }
            )
            _write_summary(summary_path, base_summary)
            raise
        completed.append(job.job_id)
        (trained if disposition == "TRAINED" else resumed).append(job.job_id)
        base_summary.update(
            {
                "completed_job_ids": list(completed),
                "trained_job_ids": list(trained),
                "resumed_job_ids": list(resumed),
                "pending_job_ids": [
                    item.job_id
                    for item in scheduled_jobs
                    if item.job_id not in set(completed)
                ],
            }
        )
        _write_summary(summary_path, base_summary)
    base_summary["status"] = "COMPLETE"
    return _write_summary(summary_path, base_summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.repro_fpo_ppo_v03.signal_fit_runner",
        description="Run and resume the real v0.3 45-fit R5/R5L schedule.",
    )
    parser.add_argument("--legacy-v0-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=_nonnegative_int, default=0)
    parser.add_argument("--shard-count", type=_positive_int, default=1)
    parser.add_argument("--train-steps", type=_positive_int, default=20_000)
    parser.add_argument("--max-jobs", type=_positive_int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as error:
        raise SystemExit(f"signal-fit runner failed: {type(error).__name__}: {error}") from error
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
