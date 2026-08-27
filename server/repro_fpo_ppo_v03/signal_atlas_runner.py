"""Minimal post-fit reader for the legacy-six-task v0.3 Signal Atlas.

This is deliberately a development runner, not a formal-authority adapter.  It
reuses the frozen 39-cell matrix, the source-fitted global canonicalizer, the
real R5/R5L checkpoint restorer, and the source-reduced/query-empirical signal
runtime.  Completed seed work and logical cells are independently atomic, so a
fresh process can continue with ``--resume`` without retraining an encoder.

The legacy source panel contains one nominal dynamics context per task.  It can
therefore measure cross-embodiment and same-embodiment/inter-goal readout, but
it cannot manufacture the missing within-task dynamics panel or pair controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.probe.dataset import load_dataset_artifact
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.schemas import FrozenProtocol
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeTransitionBank,
)
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.corro_trainers import (
    CorroOptimizationConfig,
    CorroSourceSplit,
    CorroTrainerAdapter,
)
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
    FittedRepresentation,
    RepresentationBatch,
    RepresentationManifest,
    bind_historical_random_tanh,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r2_pca_whitening,
    fit_r3_matched_random_mlp,
    restore_trained_representation,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_controls import (
    HistoricalRandomTanhSpec,
    RewardFreeShuffledNextSpec,
)
from policy_learnware_v0.v03.signal_matrix import (
    C_RF_SHUFFLED_NEXT,
    SignalCell,
    SignalCellRecord,
    SignalMatrixLedger,
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    feature_bank_from_rf_shuffled_next,
    feature_bank_from_transition_view,
    represented_bank_from_historical_random_tanh,
    run_signal_cell,
    transform_feature_banks,
)
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    TransitionBank,
    apply_transition_view,
)
from server.repro_fpo_ppo_v03.signal_fit_runner import (
    HIDDEN_DIMS,
    SHARED_OUTPUT_DIM,
    _condition_plan,
    _expected_resume_binding,
    _load_source_banks,
    _request,
    _source_split,
    _verify_completed_job,
)


SCOPE = "development/legacy-six-task-signal-atlas"
TASK_TAXONOMY = {
    "FingerSpin": ("finger", "spin"),
    "FingerTurnEasy": ("finger", "turn-easy"),
    "FingerTurnHard": ("finger", "turn-hard"),
    "WalkerStand": ("walker", "stand"),
    "WalkerWalk": ("walker", "walk"),
    "WalkerRun": ("walker", "run"),
}


class SignalAtlasRunnerError(RuntimeError):
    """The legacy assets, checkpoint set, or resume artifact is invalid."""


def signal_work_key(cell_id: str, seed: int | None) -> str:
    """Stable filename key; kept local to avoid a report/governance dependency."""

    if not isinstance(cell_id, str) or not cell_id:
        raise SignalAtlasRunnerError("work key requires a cell ID")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise SignalAtlasRunnerError("work key seed is invalid")
    return f"{cell_id.replace('::', '--')}--seed-{'NONE' if seed is None else seed}"


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _evenly_spaced(size: int, cap: int) -> np.ndarray:
    count = min(size, cap)
    if count == size:
        return np.arange(size, dtype=np.int64)
    result = np.rint(np.linspace(0, size - 1, num=count)).astype(np.int64)
    if len(np.unique(result)) != count:  # pragma: no cover - cap <= size invariant
        raise SignalAtlasRunnerError("deterministic measurement sample collided")
    return result


def _measurement_sample(
    dataset: Any,
    *,
    episodes_per_bank: int,
    transitions_per_episode: int,
) -> tuple[Mapping[str, np.ndarray], str, Mapping[str, Any]]:
    episode_indices = _evenly_spaced(dataset.episode_count, episodes_per_bank)
    selected: list[np.ndarray] = []
    episode_id: list[np.ndarray] = []
    timestep: list[np.ndarray] = []
    for new_episode, original_episode in enumerate(episode_indices.tolist()):
        start = int(dataset.episode_offsets[original_episode])
        stop = int(dataset.episode_offsets[original_episode + 1])
        local = _evenly_spaced(stop - start, transitions_per_episode)
        selected.append(start + local)
        episode_id.append(np.full(local.size, new_episode, dtype=np.int64))
        timestep.append(np.arange(local.size, dtype=np.int64))
    indices = np.concatenate(selected)
    arrays = {
        "observation": dataset.observation[indices],
        "action": dataset.action[indices],
        "reward": dataset.reward[indices],
        "next_observation": dataset.next_observation[indices],
        "terminated": dataset.terminated[indices],
        "truncated": dataset.truncated[indices],
        "episode_id": np.concatenate(episode_id),
        "timestep": np.concatenate(timestep),
    }
    full_bank = (
        episode_indices.size == dataset.episode_count
        and indices.size == dataset.transition_count
    )
    raw_digest = (
        dataset.digest
        if full_bank
        else sha256_json(
            {
                "scope": SCOPE,
                "operation": "deterministic_development_measurement_cap",
                "source_dataset_digest": dataset.digest,
                "episodes_per_bank": episodes_per_bank,
                "transitions_per_episode": transitions_per_episode,
                "selected_episode_indices": episode_indices.tolist(),
                "selected_transition_indices_digest": sha256_ndarrays(
                    {"indices": indices}
                ),
                "sampled_arrays_digest": sha256_ndarrays(arrays),
            }
        )
    )
    audit = {
        "source_dataset_digest": dataset.digest,
        "original_episode_count": dataset.episode_count,
        "original_transition_count": dataset.transition_count,
        "selected_episode_count": int(episode_indices.size),
        "selected_transition_count": int(indices.size),
        "selected_episode_indices": episode_indices.tolist(),
        "full_bank": full_bank,
        "raw_dataset_digest": raw_digest,
    }
    return arrays, raw_digest, audit


def _native_from_pair(
    manifest_path: Path,
    *,
    bank_id: str,
    role: str,
    episodes_per_bank: int,
    transitions_per_episode: int,
) -> tuple[NativeTransitionBank, str, Mapping[str, Any]]:
    npz_path = manifest_path.with_suffix(".npz")
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, npz_path)):
        raise SignalAtlasRunnerError(f"dataset pair is absent or unsafe: {manifest_path}")
    try:
        dataset, manifest = load_dataset_artifact(npz_path, manifest_path)
    except (OSError, TypeError, ValueError) as error:
        raise SignalAtlasRunnerError(f"invalid dataset {manifest_path}: {error}") from error
    if manifest_path.stem != manifest.task:
        raise SignalAtlasRunnerError(f"dataset task differs from filename: {manifest_path}")
    expected_split = {
        "source_reference_spec": "source_taskspec",
        "development_query": "target_query",
    }.get(role)
    if manifest.split != expected_split:
        raise SignalAtlasRunnerError(f"dataset split differs from role: {manifest_path}")
    arrays, raw_digest, audit = _measurement_sample(
        dataset,
        episodes_per_bank=episodes_per_bank,
        transitions_per_episode=transitions_per_episode,
    )
    schema_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-legacy-native-abi.v0",
            "task_private_id": manifest.task,
            "observation_dim": dataset.observation_dim,
            "action_dim": dataset.action_dim,
        }
    )
    probe_digest = sha256_json(
        {
            "source_dataset_digest": dataset.digest,
            "measurement_raw_dataset_digest": raw_digest,
            "probe_seed_arrays_digest": sha256_ndarrays(
                {
                    "reset_seeds": dataset.reset_seeds[
                        audit["selected_episode_indices"]
                    ],
                    "probe_seeds": dataset.probe_seeds[
                        audit["selected_episode_indices"]
                    ],
                }
            ),
        }
    )
    return (
        NativeTransitionBank(
            bank_id=bank_id,
            task_private_id=manifest.task,
            data_role=role,  # type: ignore[arg-type]
            native_schema_digest=schema_digest,
            raw_dataset_digest=raw_digest,
            observation=arrays["observation"],
            action=arrays["action"],
            reward=arrays["reward"],
            next_observation=arrays["next_observation"],
            terminated=arrays["terminated"],
            truncated=arrays["truncated"],
            episode_id=arrays["episode_id"],
            timestep=arrays["timestep"],
        ),
        probe_digest,
        audit,
    )


def _measurement_banks(
    legacy_root: Path,
    canonicalizer: GlobalCanonicalizerSpec,
    tasks: tuple[str, ...],
    expected_query_banks: int,
    episodes_per_bank: int,
    transitions_per_episode: int,
) -> tuple[
    tuple[Any, ...],
    tuple[TransitionBank, ...],
    Mapping[str, str],
    Mapping[str, Mapping[str, Any]],
]:
    native: list[NativeTransitionBank] = []
    probes: dict[str, str] = {}
    audits: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        item, probe, audit = _native_from_pair(
            legacy_root / "datasets" / "source_taskspec" / f"{task}.json",
            bank_id=f"source-{task}",
            role="source_reference_spec",
            episodes_per_bank=episodes_per_bank,
            transitions_per_episode=transitions_per_episode,
        )
        native.append(item)
        probes[item.bank_id] = probe
        audits[item.bank_id] = audit
    query_root = legacy_root / "datasets" / "target_query"
    bank_dirs = sorted(
        path
        for path in query_root.glob("bank_*")
        if path.is_dir() and not path.is_symlink()
    )
    if len(bank_dirs) != expected_query_banks:
        raise SignalAtlasRunnerError("legacy target_query bank count differs from protocol")
    for bank_dir in bank_dirs:
        for task in tasks:
            item, probe, audit = _native_from_pair(
                bank_dir / f"{task}.json",
                bank_id=f"query-{bank_dir.name}-{task}",
                role="development_query",
                episodes_per_bank=episodes_per_bank,
                transitions_per_episode=transitions_per_episode,
            )
            native.append(item)
            probes[item.bank_id] = probe
            audits[item.bank_id] = audit
    receipts = tuple(canonicalizer.transform(item) for item in native)
    banks = tuple(TransitionBank.from_canonical_batch(item.batch) for item in receipts)
    if len(receipts) != len(tasks) * (1 + len(bank_dirs)):
        raise SignalAtlasRunnerError("legacy source/query grid is incomplete")
    return receipts, banks, probes, audits


def _measurement_cap_summary(
    audits: Mapping[str, Mapping[str, Any]],
    *,
    episodes_per_bank: int,
    transitions_per_episode: int,
) -> Mapping[str, Any]:
    points = {
        bank_id: int(value["selected_transition_count"])
        for bank_id, value in sorted(audits.items())
    }
    episodes = {
        bank_id: int(value["selected_episode_count"])
        for bank_id, value in sorted(audits.items())
    }
    original_points = sum(
        int(value["original_transition_count"]) for value in audits.values()
    )
    original_episodes = sum(
        int(value["original_episode_count"]) for value in audits.values()
    )
    return {
        "development_only": True,
        "episodes_per_bank_cap": episodes_per_bank,
        "transitions_per_episode_cap": transitions_per_episode,
        "episode_selection": "equally_spaced_inclusive_endpoints",
        "transition_selection": "equally_spaced_inclusive_endpoints_per_episode",
        "transition_tuple_pairing_preserved": True,
        "bank_count": len(audits),
        "original_points_total": original_points,
        "selected_points_total": sum(points.values()),
        "selected_points_min_per_bank": min(points.values()),
        "selected_points_max_per_bank": max(points.values()),
        "original_episodes_total": original_episodes,
        "selected_episodes_total": sum(episodes.values()),
        "full_bank_count": sum(bool(value["full_bank"]) for value in audits.values()),
        "selected_points_by_bank": points,
        "selected_episodes_by_bank": episodes,
        "measurement_membership_digest": sha256_json(
            {bank_id: dict(value) for bank_id, value in sorted(audits.items())}
        ),
    }


def _identity_protocol(
    *,
    receipts: Sequence[Any],
    probes: Mapping[str, str],
    canonicalizer: GlobalCanonicalizerSpec,
    legacy_protocol: FrozenProtocol,
    condition_plan: ConditionExecutionPlan,
    optimization: CorroOptimizationConfig,
    measurement_cap: Mapping[str, Any],
) -> tuple[SignalIdentityRegistry, SignalExecutionProtocol, HistoricalRandomTanhSpec]:
    plan = build_signal_matrix_plan()
    first = TransitionBank.from_canonical_batch(receipts[0].batch)
    historical = HistoricalRandomTanhSpec.create(
        seed=condition_plan.historical_seed,
        input_dim=int(apply_transition_view(first, V_FULL_LEGACY).feature_matrix.shape[1]),
        output_dim=SHARED_OUTPUT_DIM,
    )
    measurement = sha256_json(
        {
            "scope": SCOPE,
            "legacy_protocol_id": legacy_protocol.protocol_id,
            "canonicalizer_digest": canonicalizer.canonicalizer_digest,
            "source_reduced_query_empirical": True,
            "development_measurement_cap": dict(measurement_cap),
        }
    )
    identities = []
    for receipt in receipts:
        try:
            embodiment, goal = TASK_TAXONOMY[receipt.task_private_id]
        except KeyError as error:
            raise SignalAtlasRunnerError(
                f"unregistered legacy task taxonomy: {receipt.task_private_id}"
            ) from error
        shape = canonicalizer.registry.record_for(receipt.task_private_id)
        context = f"nominal-{receipt.task_private_id}"
        identities.append(
            SignalBankIdentity.from_receipt(
                receipt,
                embodiment_id=embodiment,
                abi_contract_id=(
                    f"{embodiment}-obs{shape.observation_dim}-act{shape.action_dim}"
                ),
                goal_contract_id=f"{embodiment}-{goal}",
                dynamics_context_id="nominal",
                context_id=context,
                measurement_protocol_digest=measurement,
                probe_seed_digest=probes[receipt.bank_id],
                equivalence_class_id=receipt.task_private_id,
            )
        )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=sha256_json(TASK_TAXONOMY),
        identities=tuple(identities),
    )
    reducer_raw = dict(legacy_protocol.config["reducer"])
    reducer = ReducerConfig(
        **{
            name: reducer_raw[name]
            for name in ReducerConfig.__dataclass_fields__
            if name in reducer_raw
        }
    )
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=plan,
        historical_spec=historical,
        optimization=optimization,
    )
    protocol = SignalExecutionProtocol(
        plan_digest=str(plan.plan_digest),
        identity_registry_digest=str(registry.registry_digest),
        measurement_protocol_digest=registry.measurement_protocol_digest,
        representation_plan=representation_plan,
        condition_plan=condition_plan,
        execution_mode="DEVELOPMENT_SMOKE",
        reducer_config=reducer,
        pair_budget=int(legacy_protocol.config["kernel"]["calibration_pairs"]),
        bandwidth_seed=int(legacy_protocol.config["kernel"]["seed"]),
        block_size=2048,
        historical_seed=historical.seed,
    )
    return registry, protocol, historical


def _fit_train_steps(fits_root: Path) -> int:
    records = sorted((fits_root / "jobs").glob("job-*/job_record.json"))
    if not records:
        return 20_000
    steps = set()
    for path in records:
        value = read_json(path)
        if not isinstance(value, Mapping) or value.get("status") != "COMPLETE":
            raise SignalAtlasRunnerError(f"invalid fit job record: {path}")
        steps.add(value.get("train_steps"))
    if len(steps) != 1 or isinstance(next(iter(steps)), bool):
        raise SignalAtlasRunnerError("fit jobs do not share one train_steps value")
    return int(next(iter(steps)))


def _features(
    *,
    cell: SignalCell,
    receipts: Sequence[Any],
    banks: Sequence[TransitionBank],
    identities: Sequence[SignalBankIdentity],
    condition_plan: ConditionExecutionPlan,
) -> tuple[Any, ...]:
    result = []
    for receipt, bank, identity in zip(receipts, banks, identities, strict=True):
        if cell.condition_id == C_RF_SHUFFLED_NEXT:
            control = RewardFreeShuffledNextSpec(
                seed=condition_plan.rf_shuffled_next_seed
            ).apply(bank)
            result.append(feature_bank_from_rf_shuffled_next(receipt, identity, control))
        else:
            seed = int(condition_plan.transition_view_seeds.get(cell.condition_id, 0))
            view = apply_transition_view(bank, cell.condition_id, shuffle_seed=seed)
            result.append(feature_bank_from_transition_view(receipt, identity, view))
    return tuple(result)


def _restore_fit(
    *,
    cell: SignalCell,
    seed: int,
    fits_root: Path,
    plan: Any,
    optimization: CorroOptimizationConfig,
    train_split: CorroSourceSplit,
    validation_split: CorroSourceSplit,
) -> FittedRepresentation:
    jobs = {
        (job.cell_id, job.seed): job for job in build_optimization_fit_jobs(plan)
    }
    try:
        job = jobs[(cell.cell_id, seed)]
    except KeyError as error:
        raise SignalAtlasRunnerError(f"no frozen fit job for {cell.cell_id}, seed={seed}") from error
    trainer = CorroTrainerAdapter(
        train_split=train_split,
        validation_split=validation_split,
        optimization=optimization,
    )
    source = RepresentationBatch(
        train_split.flattened_values(), train_split.split_digest, "SOURCE_FIT"
    )
    request = _request(job, source.input_dim)
    destination = fits_root / "jobs" / f"job-{job.job_digest}"
    _verify_completed_job(
        destination,
        _expected_resume_binding(
            plan_digest=str(plan.plan_digest),
            job=job,
            optimization=optimization,
            trainer=trainer,
            source=source,
            request=request,
        ),
    )
    manifest = RepresentationManifest.from_dict(
        read_json(destination / "representation_manifest.json")
    )
    checkpoint = (destination / "checkpoint.bin").read_bytes()
    return restore_trained_representation(
        manifest=manifest,
        checkpoint_bytes=checkpoint,
        request=request,
        restorer=trainer,
        verification_source=source,
        labels=train_split.flattened_task_names(),
    )


def _fitted(
    *,
    cell: SignalCell,
    seed: int | None,
    source: RepresentationBatch,
    **restore: Any,
) -> FittedRepresentation:
    if cell.representation_id == R0_PADDED_RAW:
        return fit_r0_identity(source)
    if cell.representation_id == R1_FIXED_RANDOM_LINEAR:
        return fit_r1_random_linear(source, output_dim=SHARED_OUTPUT_DIM, seed=int(seed))
    if cell.representation_id == R2_SOURCE_PCA_WHITEN:
        return fit_r2_pca_whitening(source, output_dim=SHARED_OUTPUT_DIM)
    if cell.representation_id == R3_MATCHED_RANDOM_MLP:
        return fit_r3_matched_random_mlp(
            source,
            output_dim=SHARED_OUTPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            seed=int(seed),
        )
    if cell.representation_id in {R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR}:
        return _restore_fit(cell=cell, seed=int(seed), **restore)
    raise SignalAtlasRunnerError(f"unsupported representation: {cell.representation_id}")


def _artifact(path: Path, *, resume: bool) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if not resume:
        raise SignalAtlasRunnerError(f"output exists; use --resume: {path}")
    if path.is_symlink() or not path.is_file():
        raise SignalAtlasRunnerError(f"resume artifact is unsafe: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise SignalAtlasRunnerError(f"resume artifact is not an object: {path}")
    body = dict(value)
    supplied = body.pop("artifact_digest", None)
    if supplied != sha256_json(body):
        raise SignalAtlasRunnerError(f"resume artifact digest mismatch: {path}")
    return value


def _publish(path: Path, body: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = dict(body)
    payload["artifact_digest"] = sha256_json(payload)
    atomic_write_json(path, payload)
    return payload


def _seed_metrics(records: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    rows = [record["metric_record"]["metric_values"] for record in records]
    names = {tuple(sorted(item)) for item in rows}
    if len(names) != 1:
        raise SignalAtlasRunnerError("one cell's seed runs expose different metrics")
    return {
        name: float(np.mean([float(item[name]) for item in rows]))
        for name in next(iter(names))
    }


def _work_path(output: Path, cell: SignalCell, seed: int | None) -> Path:
    digest = sha256_json({"cell_digest": cell.cell_digest, "seed": seed})
    return output / "work" / f"work-{digest}.json"


def _cell_path(output: Path, cell: SignalCell) -> Path:
    return output / "cells" / f"cell-{cell.cell_digest}.json"


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    legacy_root = Path(args.legacy_v0_root).expanduser().resolve()
    fits_root = Path(args.signal_fits_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path, label in ((legacy_root, "legacy-v0-root"), (fits_root, "signal-fits-root")):
        if path.is_symlink() or not path.is_dir():
            raise SignalAtlasRunnerError(f"{label} must be a regular directory: {path}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "work").mkdir(exist_ok=True)
    (output / "cells").mkdir(exist_ok=True)
    if output.is_symlink():
        raise SignalAtlasRunnerError("output-dir may not be a symlink")

    legacy_protocol = FrozenProtocol.from_dict(read_json(legacy_root / "protocol" / "protocol.json"))
    tasks = tuple(legacy_protocol.config["environment"]["tasks"])
    if set(tasks) != set(TASK_TAXONOMY) or len(tasks) != 6:
        raise SignalAtlasRunnerError("legacy protocol is not the frozen six-task panel")
    (
        train_native,
        validation_native,
        canonicalizer,
        train_banks,
        validation_banks,
    ) = _load_source_banks(legacy_root)
    condition_plan = _condition_plan(train_banks[0])
    receipts, banks, probes, measurement_audits = _measurement_banks(
        legacy_root,
        canonicalizer,
        tasks,
        int(legacy_protocol.config["episodes"]["target_query_banks"]),
        int(args.episodes_per_bank),
        int(args.transitions_per_episode),
    )
    measurement_cap = _measurement_cap_summary(
        measurement_audits,
        episodes_per_bank=int(args.episodes_per_bank),
        transitions_per_episode=int(args.transitions_per_episode),
    )
    optimization = CorroOptimizationConfig(train_steps=_fit_train_steps(fits_root))
    registry, execution, historical = _identity_protocol(
        receipts=receipts,
        probes=probes,
        canonicalizer=canonicalizer,
        legacy_protocol=legacy_protocol,
        condition_plan=condition_plan,
        optimization=optimization,
        measurement_cap=measurement_cap,
    )
    identities = tuple(registry.identity_for_receipt(item) for item in receipts)
    plan = build_signal_matrix_plan()
    numeric = plan.numeric_cells
    scheduled = numeric if args.max_cells is None else numeric[: args.max_cells]
    source_count = len(tasks)
    expected = {
        receipt.bank_id: f"source-{receipt.task_private_id}"
        for receipt in receipts[source_count:]
    }
    completed_cells: list[str] = []

    for cell in scheduled:
        cell_path = _cell_path(output, cell)
        existing_cell = _artifact(cell_path, resume=bool(args.resume))
        if existing_cell is not None:
            if (
                existing_cell.get("cell_id") != cell.cell_id
                or existing_cell.get("execution_protocol_digest") != execution.protocol_digest
            ):
                raise SignalAtlasRunnerError(f"resume cell binding drifted: {cell.cell_id}")
            completed_cells.append(cell.cell_id)
            continue

        feature_banks = None
        base_views = None
        train_split = None
        validation_split = None
        if cell.representation_id == R_HIST_RANDOM_TANH:
            base_views = tuple(apply_transition_view(bank, V_FULL_LEGACY) for bank in banks)
            seed_schedule: tuple[int | None, ...] = (historical.seed,)
        else:
            feature_banks = _features(
                cell=cell,
                receipts=receipts,
                banks=banks,
                identities=identities,
                condition_plan=condition_plan,
            )
            seed_schedule = execution.expected_evaluation_seeds(cell)
            if cell.representation_id in {
                R2_SOURCE_PCA_WHITEN,
                R5_VIEW_SPECIFIC_CORRO_REFIT,
                R5L_SUPERVISED_LINEAR,
            }:
                train_split = _source_split(
                    role="source_representation_train",
                    native_banks=train_native,
                    transition_banks=train_banks,
                    condition_id=cell.condition_id,
                    condition_plan=condition_plan,
                )
                source_fit = RepresentationBatch(
                    train_split.flattened_values(),
                    train_split.split_digest,
                    "SOURCE_FIT",
                )
                if cell.representation_id in {
                    R5_VIEW_SPECIFIC_CORRO_REFIT,
                    R5L_SUPERVISED_LINEAR,
                }:
                    validation_split = _source_split(
                        role="source_representation_validation",
                        native_banks=validation_native,
                        transition_banks=validation_banks,
                        condition_id=cell.condition_id,
                        condition_plan=condition_plan,
                    )
            else:
                source_fit = RepresentationBatch(
                    values=np.concatenate(
                        [item.values for item in feature_banks[:source_count]], axis=0
                    ),
                    dataset_digest=sha256_json(
                        {
                            "scope": SCOPE,
                            "non_data_fitted_source_features": [
                                item.feature_bank_digest
                                for item in feature_banks[:source_count]
                            ],
                        }
                    ),
                    role="SOURCE_FIT",
                )

        work_records = []
        for seed in seed_schedule:
            work_path = _work_path(output, cell, seed)
            existing = _artifact(work_path, resume=bool(args.resume))
            if existing is not None:
                if (
                    existing.get("work_key") != signal_work_key(cell.cell_id, seed)
                    or existing.get("execution_protocol_digest") != execution.protocol_digest
                ):
                    raise SignalAtlasRunnerError(f"resume work binding drifted: {cell.cell_id}, {seed}")
                work_records.append(existing)
                continue
            if cell.representation_id == R_HIST_RANDOM_TANH:
                assert base_views is not None
                source = RepresentationBatch(
                    values=np.concatenate(
                        [item.feature_matrix for item in base_views[:source_count]], axis=0
                    ),
                    dataset_digest=sha256_json(
                        {
                            "scope": SCOPE,
                            "historical_source": [
                                item.view_digest for item in base_views[:source_count]
                            ],
                        }
                    ),
                    role="SOURCE_FIT",
                )
                fitted = bind_historical_random_tanh(source, spec=historical)
                represented = tuple(
                    represented_bank_from_historical_random_tanh(
                        receipt,
                        identity,
                        base,
                        historical.apply(bank),
                        fitted,
                    )
                    for receipt, identity, base, bank in zip(
                        receipts, identities, base_views, banks, strict=True
                    )
                )
            else:
                assert feature_banks is not None
                fitted = _fitted(
                    cell=cell,
                    seed=seed,
                    source=source_fit,
                    fits_root=fits_root,
                    plan=plan,
                    optimization=optimization,
                    train_split=train_split,
                    validation_split=validation_split,
                )
                represented = transform_feature_banks(fitted, feature_banks)
            run_result = run_signal_cell(
                plan=plan,
                cell=cell,
                source_banks=represented[:source_count],
                query_banks=represented[source_count:],
                expected_source_by_query=expected,
                identity_registry=registry,
                execution_protocol=execution,
            )
            work_records.append(
                _publish(
                    work_path,
                    {
                        "scope": SCOPE,
                        "formal_run_authorized": False,
                        "work_key": signal_work_key(cell.cell_id, seed),
                        "cell_id": cell.cell_id,
                        "evaluation_seed": seed,
                        "execution_protocol_digest": execution.protocol_digest,
                        "run": run_result.to_dict(),
                        "kernel_protocol": run_result.kernel_protocol.to_dict(),
                        "metric_record": run_result.metric_record.to_dict(),
                        "diagnostics": run_result.diagnostics.to_private_dict(),
                    },
                )
            )
        record = SignalCellRecord(
            plan_digest=str(plan.plan_digest),
            cell_id=cell.cell_id,
            cell_digest=str(cell.cell_digest),
            status="COMPUTED",
            metrics=_seed_metrics(work_records),
            numeric_artifact_digest=sha256_json(
                {
                    "schema": "policy-learnware.v03-seed-aggregate.v0",
                    "cell_digest": cell.cell_digest,
                    "seed_run_digests": sorted(item["run"]["run_digest"] for item in work_records),
                }
            ),
        )
        _publish(
            cell_path,
            {
                "scope": SCOPE,
                "formal_run_authorized": False,
                "cell_id": cell.cell_id,
                "execution_protocol_digest": execution.protocol_digest,
                "seed_work_artifact_digests": [item["artifact_digest"] for item in work_records],
                "record": record.to_dict(),
            },
        )
        completed_cells.append(cell.cell_id)
        atomic_write_json(
            output / "progress.json",
            {
                "scope": SCOPE,
                "status": "RUNNING",
                "completed_numeric_cell_count": len(completed_cells),
                "scheduled_numeric_cell_count": len(scheduled),
                "completed_cell_ids": completed_cells,
            },
            overwrite=True,
        )

    all_cell_artifacts = {
        cell.cell_id: _artifact(_cell_path(output, cell), resume=True)
        for cell in numeric
        if _cell_path(output, cell).exists()
    }
    complete = len(all_cell_artifacts) == plan.numeric_cell_count
    ledger_digest = None
    if complete:
        records_by_id = {
            cell_id: SignalCellRecord.from_dict(value["record"])
            for cell_id, value in all_cell_artifacts.items()
            if value is not None
        }
        records = []
        for cell in plan.cells:
            if cell.applicability == "STRUCTURAL_NA":
                records.append(
                    SignalCellRecord(
                        plan_digest=str(plan.plan_digest),
                        cell_id=cell.cell_id,
                        cell_digest=str(cell.cell_digest),
                        status="STRUCTURAL_NA",
                        metrics=None,
                        numeric_artifact_digest=None,
                    )
                )
            else:
                records.append(records_by_id[cell.cell_id])
        ledger = SignalMatrixLedger(plan=plan, records=tuple(records))
        atomic_write_json(output / "ledger.json", ledger.to_dict(), overwrite=True)
        ledger_digest = ledger.ledger_digest

    summary = {
        "status": (
            "LEGACY_SIX_TASK_SIGNAL_ATLAS_COMPLETE"
            if complete
            else "LEGACY_SIX_TASK_SIGNAL_ATLAS_PARTIAL"
        ),
        "scope": SCOPE,
        "formal_run_authorized": False,
        "signal_matrix_plan_digest": plan.plan_digest,
        "execution_protocol_digest": execution.protocol_digest,
        "identity_registry_digest": registry.registry_digest,
        "canonicalizer_digest": canonicalizer.canonicalizer_digest,
        "logical_cell_count": plan.logical_cell_count,
        "numeric_cell_count": plan.numeric_cell_count,
        "structural_na_count": plan.structural_na_count,
        "completed_numeric_cell_count": len(all_cell_artifacts),
        "ledger_digest": ledger_digest,
        "source_task_count": source_count,
        "query_bank_count": len(receipts) - source_count,
        "task_private_ids": list(tasks),
        "supported_readouts": ["cross_embodiment", "same_embodiment_inter_goal"],
        "unavailable_from_legacy_six_task_inputs": [
            "within_task_dynamics: source_taskspec contains nominal only",
            "C_SCHEMA_COLLISION and C_EXACT_REPEAT: pair/bank controls are separate from the 39 rows",
            "frozen legacy MLP A-table: this runner is the post-fit S-table reader",
            "formal publication: external authority and formal CP0/CP2 banks are not supplied",
        ],
        "source_kme_mode": "REDUCED",
        "query_kme_mode": "EMPIRICAL",
        "development_measurement_cap": dict(measurement_cap),
        "full_bank_measurement": measurement_cap["full_bank_count"]
        == measurement_cap["bank_count"],
        "signal_fit_train_steps": optimization.train_steps,
    }
    atomic_write_json(output / "summary.json", summary, overwrite=True)
    atomic_write_json(
        output / "progress.json",
        {
            "scope": SCOPE,
            "status": "COMPLETE" if complete else "PARTIAL",
            "completed_numeric_cell_count": len(all_cell_artifacts),
            "scheduled_numeric_cell_count": len(scheduled),
            "completed_cell_ids": sorted(all_cell_artifacts),
        },
        overwrite=True,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read completed signal fits into the real legacy-six-task development Atlas"
    )
    parser.add_argument("--legacy-v0-root", required=True)
    parser.add_argument("--signal-fits-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cells", type=_positive_int)
    parser.add_argument("--episodes-per-bank", type=_positive_int, default=4)
    parser.add_argument("--transitions-per-episode", type=_positive_int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as error:
        raise SystemExit(
            f"signal-atlas runner failed: {type(error).__name__}: {error}"
        ) from error
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
