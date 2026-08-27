"""Minimal v0.3 dynamics-Signal runner for collected context-bank NPZ files.

The context index is a JSON object with a ``contexts`` list.  Each row has
``bank_id``, ``role``, ``task_id``, ``embodiment_id``, ``goal_id``,
``dynamics_id``, ``context_id`` and ``npz``.  Roles are ``source`` or a native
query role; query rows also name ``expected_source_bank_id``.  NPZ files only
need the ordinary transition arrays plus ``episode_offsets``.

A source NPZ is stored once.  Episodes are deterministically split into
task-SupCon train/validation and two non-overlapping reference banks.  The
first reference bank enters the 30-source index; the second is the independent
same-context repeat used for C_EXACT_REPEAT.  Anchors are grouped by their six
base task IDs during representation fitting, while their separate context IDs
remain visible to RKME ranking and dynamics metrics.

The runner schedules only the frozen dynamics-facing subset of the 39-cell
plan.  Fit jobs and cell/seed work are immutable, independently resumable and
shardable.  JSONL outputs are regenerable long-form indexes, not a new contract
or authority layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    read_npz,
)
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.corro_trainers import (
    CorroOptimizationConfig,
    CorroSourceSplit,
    CorroTaskDataset,
    CorroTrainerAdapter,
)
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    FittedRepresentation,
    RepresentationBatch,
    RepresentationManifest,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r2_pca_whitening,
    fit_r3_matched_random_mlp,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
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
    SignalFitJob,
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    feature_bank_from_rf_shuffled_next,
    feature_bank_from_transition_view,
    run_signal_cell,
    transform_feature_banks,
)
from policy_learnware_v0.v03.transition_views import (
    V_DELTA_ONLY,
    V_FULL_LEGACY,
    V_REWARD_FREE_TRANSITION,
    V_STATE_ACTION,
    V_STATE_ONLY,
    TransitionBank,
    apply_transition_view,
)
from server.repro_fpo_ppo_v03.signal_fit_runner import (
    HIDDEN_DIMS,
    SHARED_OUTPUT_DIM,
    _condition_plan,
    _expected_resume_binding,
    _job_directory,
    _publish_completed_job,
    _request,
    _verify_completed_job,
)


SCOPE = "development/v02-dynamics-signal"
CORE_ARRAYS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "episode_offsets",
)
QUERY_ROLES = frozenset({"development_query", "confirmatory_query"})
DYNAMICS_CONDITIONS = frozenset(
    {
        V_FULL_LEGACY,
        V_REWARD_FREE_TRANSITION,
        V_STATE_ONLY,
        V_STATE_ACTION,
        V_DELTA_ONLY,
        C_RF_SHUFFLED_NEXT,
    }
)


class SignalBankRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Inputs:
    index_digest: str
    source_rows: tuple[Mapping[str, Any], ...]
    query_rows: tuple[Mapping[str, Any], ...]
    train_native: tuple[NativeTransitionBank, ...]
    validation_native: tuple[NativeTransitionBank, ...]
    train_banks: tuple[TransitionBank, ...]
    validation_banks: tuple[TransitionBank, ...]
    receipts: tuple[Any, ...]
    banks: tuple[TransitionBank, ...]
    identities: tuple[SignalBankIdentity, ...]
    registry: SignalIdentityRegistry
    canonicalizer: GlobalCanonicalizerSpec
    condition_plan: Any
    measurement: Mapping[str, Any]

    @property
    def source_count(self) -> int:
        return len(self.source_rows)

    @property
    def query_count(self) -> int:
        return len(self.query_rows)

    @property
    def repeat_count(self) -> int:
        return len(self.source_rows)


def _int(value: str, *, zero: bool = False) -> int:
    result = int(value)
    if result < 0 or (result == 0 and not zero):
        raise argparse.ArgumentTypeError("invalid integer")
    return result


def _positive(value: str) -> int:
    return _int(value)


def _nonnegative(value: str) -> int:
    return _int(value, zero=True)


def _load_index(path: Path) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    raw = read_json(path)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("contexts"), list):
        raise SignalBankRunnerError("context index must contain a contexts list")
    required = {
        "bank_id",
        "role",
        "task_id",
        "embodiment_id",
        "goal_id",
        "dynamics_id",
        "context_id",
        "axis_id",
        "factor_id",
        "npz",
    }
    rows = []
    for value in raw["contexts"]:
        if not isinstance(value, Mapping) or any(
            not isinstance(value.get(name), str) or not value[name]
            for name in required
        ):
            raise SignalBankRunnerError("context row is missing a required string")
        row = dict(value)
        if isinstance(row.get("factor_value"), bool) or not isinstance(
            row.get("factor_value"), (int, float)
        ) or not np.isfinite(float(row["factor_value"])):
            raise SignalBankRunnerError("context factor_value must be finite numeric")
        row["factor_value"] = float(row["factor_value"])
        role = row["role"]
        if role not in {"source", *QUERY_ROLES}:
            raise SignalBankRunnerError(f"unsupported context role: {role}")
        if role in QUERY_ROLES and not isinstance(
            row.get("expected_source_bank_id"), str
        ):
            raise SignalBankRunnerError("query lacks expected_source_bank_id")
        candidate = Path(row["npz"]).expanduser()
        row["npz"] = str(
            (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        )
        row.setdefault("equivalence_class_id", row["context_id"])
        rows.append(row)
    rows.sort(key=lambda item: item["bank_id"])
    ids = [row["bank_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SignalBankRunnerError("context bank IDs must be unique")
    sources = {row["bank_id"] for row in rows if row["role"] == "source"}
    queries = [row for row in rows if row["role"] in QUERY_ROLES]
    if len(sources) < 2 or not queries:
        raise SignalBankRunnerError("at least two sources and one query are required")
    if any(row["expected_source_bank_id"] not in sources for row in queries):
        raise SignalBankRunnerError("query expected source is absent")
    return sha256_json(rows), tuple(rows)


def _load_arrays(row: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    path = Path(row["npz"])
    if path.is_symlink() or not path.is_file():
        raise SignalBankRunnerError(f"bank NPZ is absent or symlinked: {path}")
    arrays = read_npz(path)
    missing = set(CORE_ARRAYS) - set(arrays)
    if missing:
        raise SignalBankRunnerError(f"{row['bank_id']}: missing {sorted(missing)}")
    core = {name: arrays[name] for name in CORE_ARRAYS}
    TransitionBank(**core)
    return {**core, **{name: arrays[name] for name in ("reset_seeds", "probe_seeds") if name in arrays}}


def _evenly_spaced(size: int, cap: int) -> np.ndarray:
    count = min(int(size), int(cap))
    if count == size:
        return np.arange(size, dtype=np.int64)
    result = np.rint(np.linspace(0, size - 1, num=count)).astype(np.int64)
    if len(np.unique(result)) != count:
        raise SignalBankRunnerError("deterministic measurement sample collided")
    return result


def _subset(
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    episodes: np.ndarray,
    *,
    bank_id: str,
    role: str,
    episode_cap: int | None = None,
    transition_cap: int | None = None,
) -> tuple[NativeTransitionBank, str, Mapping[str, Any]]:
    offsets = np.asarray(arrays["episode_offsets"], dtype=np.int64)
    if (episode_cap is None) != (transition_cap is None):
        raise SignalBankRunnerError("measurement caps must be set together")
    candidate_episodes = np.asarray(episodes, dtype=np.int64)
    selected_episodes = (
        candidate_episodes
        if episode_cap is None
        else candidate_episodes[_evenly_spaced(len(candidate_episodes), episode_cap)]
    )
    pieces = []
    for episode in selected_episodes:
        start, stop = int(offsets[episode]), int(offsets[episode + 1])
        local = (
            np.arange(stop - start, dtype=np.int64)
            if transition_cap is None
            else _evenly_spaced(stop - start, transition_cap)
        )
        pieces.append(start + local)
    indices = np.concatenate(pieces)
    new_offsets = np.concatenate(
        [[0], np.cumsum([len(piece) for piece in pieces], dtype=np.int64)]
    )
    episode_id = np.concatenate(
        [np.full(len(piece), i, dtype=np.int64) for i, piece in enumerate(pieces)]
    )
    timestep = np.concatenate(
        [np.arange(len(piece), dtype=np.int64) for piece in pieces]
    )
    core = {
        name: np.asarray(arrays[name])[indices]
        for name in CORE_ARRAYS
        if name != "episode_offsets"
    }
    core["episode_offsets"] = new_offsets
    raw_digest = sha256_ndarrays(core)
    observation, action = core["observation"], core["action"]
    schema_digest = sha256_json(
        {
            "task_id": row["task_id"],
            "observation_dim": int(observation.shape[1]),
            "action_dim": int(action.shape[1]),
        }
    )
    seed_arrays = {
        name: np.asarray(arrays[name])[selected_episodes]
        for name in ("reset_seeds", "probe_seeds")
        if name in arrays
    }
    probe_digest = sha256_json(
        {
            "raw_digest": raw_digest,
            "episodes": selected_episodes.tolist(),
            "transition_indices_digest": sha256_ndarrays({"indices": indices}),
            "seeds": None if not seed_arrays else sha256_ndarrays(seed_arrays),
        }
    )
    bank = NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=row["task_id"],
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=schema_digest,
        raw_dataset_digest=raw_digest,
        observation=observation,
        action=action,
        reward=core["reward"],
        next_observation=core["next_observation"],
        terminated=core["terminated"],
        truncated=core["truncated"],
        episode_id=episode_id,
        timestep=timestep,
    )
    audit = {
        "bank_id": bank_id,
        "candidate_episode_count": int(len(candidate_episodes)),
        "selected_episode_count": int(len(selected_episodes)),
        "selected_transition_count": int(len(indices)),
        "episode_cap": episode_cap,
        "transitions_per_episode_cap": transition_cap,
        "selected_episode_indices": selected_episodes.tolist(),
        "selected_transition_indices_digest": sha256_ndarrays({"indices": indices}),
    }
    return bank, probe_digest, audit


def _source_partition(
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(arrays["episode_offsets"]) - 1
    train = max(1, int(np.floor(count * train_fraction)))
    validation = max(1, int(np.floor(count * validation_fraction)))
    if count - train - validation < 4:
        raise SignalBankRunnerError(
            f"{row['bank_id']}: source needs four reference episodes after train/validation"
        )
    digest = sha256_ndarrays({name: arrays[name] for name in CORE_ARRAYS})
    order = np.random.default_rng(seed ^ int(digest[:16], 16)).permutation(count)
    held = order[train + validation :]
    midpoint = len(held) // 2
    if midpoint < 2 or len(held) - midpoint < 2:
        raise SignalBankRunnerError(f"{row['bank_id']}: repeat split is too small")
    return tuple(
        np.sort(item)
        for item in (
            order[:train],
            order[train : train + validation],
            held[:midpoint],
            held[midpoint:],
        )
    )  # type: ignore[return-value]


def prepare_signal_inputs(
    index_path: Path,
    train_fraction: float,
    validation_fraction: float,
    split_seed: int,
    measurement_episodes: int = 16,
    measurement_transitions: int = 64,
) -> Inputs:
    index_digest, rows = _load_index(index_path)
    sources = tuple(row for row in rows if row["role"] == "source")
    queries = tuple(row for row in rows if row["role"] in QUERY_ROLES)
    train_native, validation_native, source_native, repeat_native = [], [], [], []
    probes: dict[str, str] = {}
    measurement_audits: list[Mapping[str, Any]] = []
    repeat_rows = []
    for row in sources:
        arrays = _load_arrays(row)
        train_ep, validation_ep, source_ep, repeat_ep = _source_partition(
            row, arrays, train_fraction, validation_fraction, split_seed
        )
        for target, episodes, bank_id, role, measured in (
            (train_native, train_ep, f"{row['bank_id']}-train", "source_representation_train", False),
            (validation_native, validation_ep, f"{row['bank_id']}-validation", "source_representation_validation", False),
            (source_native, source_ep, row["bank_id"], "source_reference_spec", True),
            (repeat_native, repeat_ep, f"{row['bank_id']}-repeat", "development_query", True),
        ):
            bank, probe, audit = _subset(
                row,
                arrays,
                episodes,
                bank_id=bank_id,
                role=role,
                episode_cap=measurement_episodes if measured else None,
                transition_cap=measurement_transitions if measured else None,
            )
            target.append(bank)
            probes[bank_id] = probe
            if measured:
                measurement_audits.append(audit)
        repeat_rows.append(
            {
                **row,
                "bank_id": f"{row['bank_id']}-repeat",
                "role": "development_query",
                "expected_source_bank_id": row["bank_id"],
            }
        )
    query_native = []
    for row in queries:
        arrays = _load_arrays(row)
        episodes = np.arange(len(arrays["episode_offsets"]) - 1, dtype=np.int64)
        bank, probe, audit = _subset(
            row,
            arrays,
            episodes,
            bank_id=row["bank_id"],
            role=row["role"],
            episode_cap=measurement_episodes,
            transition_cap=measurement_transitions,
        )
        query_native.append(bank)
        probes[row["bank_id"]] = probe
        measurement_audits.append(audit)
    registry = NativeShapeRegistry.from_source_banks((*train_native, *validation_native))
    canonicalizer = GlobalCanonicalizerSpec(
        registry,
        fit_global_normalizer((*train_native, *validation_native), registry),
    )
    train_receipts = tuple(canonicalizer.transform(bank) for bank in train_native)
    validation_receipts = tuple(canonicalizer.transform(bank) for bank in validation_native)
    train_banks = tuple(TransitionBank.from_canonical_batch(item.batch) for item in train_receipts)
    validation_banks = tuple(
        TransitionBank.from_canonical_batch(item.batch) for item in validation_receipts
    )
    measured_native = tuple((*source_native, *query_native, *repeat_native))
    receipts = tuple(canonicalizer.transform(bank) for bank in measured_native)
    banks = tuple(TransitionBank.from_canonical_batch(item.batch) for item in receipts)
    metadata = {
        row["bank_id"]: row for row in (*sources, *queries, *repeat_rows)
    }
    measurement_digest = sha256_json(
        {
            "scope": SCOPE,
            "index": index_digest,
            "canonicalizer": canonicalizer.canonicalizer_digest,
            "split": [train_fraction, validation_fraction, split_seed],
            "episode_cap": measurement_episodes,
            "transitions_per_episode_cap": measurement_transitions,
            "membership": list(measurement_audits),
        }
    )
    measurement = {
        "episode_cap": measurement_episodes,
        "transitions_per_episode_cap": measurement_transitions,
        "selection": "evenly_spaced_episodes/evenly_spaced_transitions",
        "bank_count": len(measurement_audits),
        "point_count_min": min(item["selected_transition_count"] for item in measurement_audits),
        "point_count_max": max(item["selected_transition_count"] for item in measurement_audits),
        "membership_digest": sha256_json(list(measurement_audits)),
        "banks": list(measurement_audits),
        "full_collector_npz_preserved": True,
    }
    identities = []
    for receipt in receipts:
        row = metadata[receipt.bank_id]
        shape = registry.record_for(row["task_id"])
        identities.append(
            SignalBankIdentity.from_receipt(
                receipt,
                embodiment_id=row["embodiment_id"],
                abi_contract_id=f"{row['embodiment_id']}-obs{shape.observation_dim}-act{shape.action_dim}",
                goal_contract_id=row["goal_id"],
                dynamics_context_id=row["dynamics_id"],
                context_id=row["context_id"],
                measurement_protocol_digest=measurement_digest,
                probe_seed_digest=probes[receipt.bank_id],
                equivalence_class_id=row["equivalence_class_id"],
            )
        )
    identity_registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=sha256_json(
            [{name: row[name] for name in ("bank_id", "task_id", "embodiment_id", "goal_id", "dynamics_id", "context_id")} for row in rows]
        ),
        identities=tuple(identities),
    )
    return Inputs(
        index_digest,
        sources,
        queries,
        tuple(train_native),
        tuple(validation_native),
        train_banks,
        validation_banks,
        receipts,
        banks,
        tuple(identity_registry.identity_for_receipt(item) for item in receipts),
        identity_registry,
        canonicalizer,
        _condition_plan(train_banks[0]),
        measurement,
    )


def _offsets(banks: Sequence[TransitionBank]) -> np.ndarray:
    result = [0]
    for bank in banks:
        result.extend((result[-1] + bank.episode_offsets[1:]).tolist())
    return np.asarray(result, dtype=np.int64)


def _source_split(inputs: Inputs, condition: str, validation: bool) -> CorroSourceSplit:
    native = inputs.validation_native if validation else inputs.train_native
    banks = inputs.validation_banks if validation else inputs.train_banks
    grouped: dict[str, list[TransitionBank]] = {}
    for item, bank in zip(native, banks, strict=True):
        grouped.setdefault(item.task_private_id, []).append(bank)
    tasks = []
    for task_id, task_banks in sorted(grouped.items()):
        values = []
        for bank in task_banks:
            if condition == C_RF_SHUFFLED_NEXT:
                values.append(
                    RewardFreeShuffledNextSpec(
                        seed=inputs.condition_plan.rf_shuffled_next_seed
                    ).apply(bank).feature_matrix
                )
            else:
                values.append(
                    apply_transition_view(
                        bank,
                        condition,
                        shuffle_seed=int(
                            inputs.condition_plan.transition_view_seeds.get(condition, 0)
                        ),
                    ).feature_matrix
                )
        tasks.append(
            CorroTaskDataset(
                task_id,
                np.concatenate(values),
                _offsets(task_banks),
            )
        )
    role = "source_representation_validation" if validation else "source_representation_train"
    return CorroSourceSplit(role, tuple(tasks))  # type: ignore[arg-type]


def _cells() -> tuple[SignalCell, ...]:
    return tuple(
        cell
        for cell in build_signal_matrix_plan().numeric_cells
        if cell.condition_id in DYNAMICS_CONDITIONS
    )


def _jobs() -> tuple[SignalFitJob, ...]:
    cell_ids = {cell.cell_id for cell in _cells()}
    return tuple(
        job
        for job in build_optimization_fit_jobs(build_signal_matrix_plan())
        if job.cell_id in cell_ids
    )


def _assigned(values: Sequence[Any], index: int, count: int) -> tuple[Any, ...]:
    return tuple(value for position, value in enumerate(values) if position % count == index)


def _fit_context(inputs: Inputs, condition: str, optimization: CorroOptimizationConfig):
    train = _source_split(inputs, condition, False)
    validation = _source_split(inputs, condition, True)
    trainer = CorroTrainerAdapter(train, validation, optimization)
    source = RepresentationBatch(train.flattened_values(), train.split_digest, "SOURCE_FIT")
    return train, trainer, source


def _fit_job(
    inputs: Inputs,
    output: Path,
    job: SignalFitJob,
    optimization: CorroOptimizationConfig,
    resume: bool,
) -> Mapping[str, Any]:
    train, trainer, source = _fit_context(inputs, job.condition_id, optimization)
    request = _request(job, source.input_dim)
    expected = _expected_resume_binding(
        plan_digest=str(build_signal_matrix_plan().plan_digest),
        job=job,
        optimization=optimization,
        trainer=trainer,
        source=source,
        request=request,
    )
    root = output / "fits"
    destination = _job_directory(root, job)
    if destination.exists():
        if not resume:
            raise SignalBankRunnerError(f"fit exists; use --resume: {destination}")
        record = _verify_completed_job(destination, expected)
        disposition = "RESUMED"
    else:
        common = dict(
            labels=train.flattened_task_names(),
            trainer=trainer,
            objective_digest=request.objective_digest,
            seed=job.seed,
            output_dim=SHARED_OUTPUT_DIM,
        )
        fitted = (
            fit_r5_corro_style(source, hidden_dims=HIDDEN_DIMS, **common)
            if job.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
            else fit_r5l_supervised_linear(source, **common)
        )
        record = _publish_completed_job(
            output_dir=root,
            plan_digest=str(build_signal_matrix_plan().plan_digest),
            job=job,
            optimization=optimization,
            trainer=trainer,
            source=source,
            request=request,
            fitted=fitted,
        )
        disposition = "TRAINED"
    return {
        "row_id": job.job_digest,
        "job_id": job.job_id,
        "condition_id": job.condition_id,
        "representation_id": job.representation_id,
        "seed": job.seed,
        "status": "COMPLETE",
        "disposition": disposition,
        "checkpoint": str(destination / "checkpoint.bin"),
        "checkpoint_sha256": record["checkpoint_file_sha256"],
    }


def _restore(
    inputs: Inputs,
    output: Path,
    cell: SignalCell,
    seed: int,
    optimization: CorroOptimizationConfig,
) -> FittedRepresentation:
    job = {(item.cell_id, item.seed): item for item in _jobs()}[(cell.cell_id, seed)]
    train, trainer, source = _fit_context(inputs, cell.condition_id, optimization)
    request = _request(job, source.input_dim)
    destination = _job_directory(output / "fits", job)
    _verify_completed_job(
        destination,
        _expected_resume_binding(
            plan_digest=str(build_signal_matrix_plan().plan_digest),
            job=job,
            optimization=optimization,
            trainer=trainer,
            source=source,
            request=request,
        ),
    )
    return restore_trained_representation(
        manifest=RepresentationManifest.from_dict(read_json(destination / "representation_manifest.json")),
        checkpoint_bytes=(destination / "checkpoint.bin").read_bytes(),
        request=request,
        restorer=trainer,
        verification_source=source,
        labels=train.flattened_task_names(),
    )


def _features(inputs: Inputs, cell: SignalCell) -> tuple[Any, ...]:
    result = []
    for receipt, bank, identity in zip(inputs.receipts, inputs.banks, inputs.identities, strict=True):
        if cell.condition_id == C_RF_SHUFFLED_NEXT:
            control = RewardFreeShuffledNextSpec(
                seed=inputs.condition_plan.rf_shuffled_next_seed
            ).apply(bank)
            result.append(feature_bank_from_rf_shuffled_next(receipt, identity, control))
        else:
            view = apply_transition_view(
                bank,
                cell.condition_id,
                shuffle_seed=int(inputs.condition_plan.transition_view_seeds.get(cell.condition_id, 0)),
            )
            result.append(feature_bank_from_transition_view(receipt, identity, view))
    return tuple(result)


def _representation(
    inputs: Inputs,
    output: Path,
    cell: SignalCell,
    seed: int | None,
    source: RepresentationBatch,
    optimization: CorroOptimizationConfig,
) -> FittedRepresentation:
    if cell.representation_id == R0_PADDED_RAW:
        return fit_r0_identity(source)
    if cell.representation_id == R1_FIXED_RANDOM_LINEAR:
        return fit_r1_random_linear(source, output_dim=SHARED_OUTPUT_DIM, seed=int(seed))
    if cell.representation_id == R2_SOURCE_PCA_WHITEN:
        return fit_r2_pca_whitening(source, output_dim=SHARED_OUTPUT_DIM)
    if cell.representation_id == R3_MATCHED_RANDOM_MLP:
        return fit_r3_matched_random_mlp(
            source, output_dim=SHARED_OUTPUT_DIM, hidden_dims=HIDDEN_DIMS, seed=int(seed)
        )
    return _restore(inputs, output, cell, int(seed), optimization)


def _protocol(inputs: Inputs, args: argparse.Namespace, optimization: CorroOptimizationConfig):
    plan = build_signal_matrix_plan()
    width = apply_transition_view(inputs.train_banks[0], V_FULL_LEGACY).feature_matrix.shape[1]
    historical = HistoricalRandomTanhSpec.create(seed=0, input_dim=width, output_dim=SHARED_OUTPUT_DIM)
    return SignalExecutionProtocol(
        plan_digest=str(plan.plan_digest),
        identity_registry_digest=str(inputs.registry.registry_digest),
        measurement_protocol_digest=inputs.registry.measurement_protocol_digest,
        representation_plan=RepresentationExecutionPlan.create(
            signal_plan=plan, historical_spec=historical, optimization=optimization
        ),
        condition_plan=inputs.condition_plan,
        execution_mode="DEVELOPMENT_SMOKE",
        reducer_config=ReducerConfig(
            support_budget=args.support_budget,
            support_steps=args.support_steps,
            kmeans_steps=args.kmeans_steps,
            optimizer_backend=args.reducer_backend,
        ),
        pair_budget=args.pair_budget,
        bandwidth_seed=args.bandwidth_seed,
        block_size=args.block_size,
    )


def _work_path(output: Path, cell: SignalCell, seed: int | None) -> Path:
    return output / "atlas" / "work" / f"work-{sha256_json({'cell': cell.cell_digest, 'seed': seed})}.json"


def _repeat_rows(repeat_run: Any, source_count: int) -> list[Mapping[str, Any]]:
    rows = repeat_run.metric_record.rows
    result = []
    for query_id in sorted({row.query_bank_id for row in rows}):
        matches = [
            row for row in rows
            if row.query_bank_id == query_id
            and row.query_context_id == row.source_context_id
            and row.query_task_id == row.source_task_id
            and row.query_dynamics_context_id == row.source_dynamics_context_id
        ]
        if len(matches) != 1:
            raise SignalBankRunnerError(f"exact repeat pair is not unique for {query_id}")
        direct = matches[0]
        query_id, source_id = direct.query_bank_id, direct.source_bank_id
        between = [
            row.distance for row in rows
            if row.query_bank_id == query_id
            and row.source_task_id == direct.query_task_id
            and row.source_goal_contract_id == direct.query_goal_contract_id
            and row.source_embodiment_id == direct.query_embodiment_id
            and row.source_abi_contract_id == direct.query_abi_contract_id
            and row.source_dynamics_context_id != direct.query_dynamics_context_id
        ]
        between_mean = None if not between else float(np.mean(between))
        ratio = None if between_mean is None or direct.distance == 0.0 else between_mean / direct.distance
        result.append(
            {
                "row_id": sha256_json(
                    {"run": repeat_run.run_digest, "query": query_id, "source": source_id}
                ),
                "cell_id": repeat_run.cell_id,
                "representation_id": repeat_run.metric_record.representation_id,
                "seed": repeat_run.evaluation_seed,
                "query_bank_id": query_id,
                "source_bank_id": source_id,
                "direct_repeat_mmd": direct.distance,
                "between_dynamics_mean_mmd": between_mean,
                "between_repeat_ratio": ratio,
                "ratio_kind": (
                    "NO_BETWEEN_DYNAMICS"
                    if between_mean is None
                    else "INFINITE_ZERO_NOISE_FLOOR"
                    if direct.distance == 0.0
                    else "FINITE"
                ),
            }
        )
    if len(result) != source_count:
        raise SignalBankRunnerError("exact repeat coverage differs from source count")
    return result


def _rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _interpolation_metrics(
    metric_record: Any,
    source_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, float]]:
    """Read bracket/order evidence without redefining an interpolation as an anchor."""

    by_query: dict[str, list[Any]] = {}
    for row in metric_record.rows:
        by_query.setdefault(row.query_bank_id, []).append(row)
    details = []
    for query in query_rows:
        q = float(query["factor_value"])
        axis_sources = [
            source
            for source in source_rows
            if source["task_id"] == query["task_id"]
            and (
                source["axis_id"] == query["axis_id"]
                or np.isclose(float(source["factor_value"]), 1.0)
            )
        ]
        lower = [row for row in axis_sources if float(row["factor_value"]) < q]
        upper = [row for row in axis_sources if float(row["factor_value"]) > q]
        if not lower or not upper or q <= 0 or any(float(row["factor_value"]) <= 0 for row in axis_sources):
            continue
        left = max(lower, key=lambda row: float(row["factor_value"]))
        right = min(upper, key=lambda row: float(row["factor_value"]))
        bracket = {left["bank_id"], right["bank_id"]}
        ranked = sorted(by_query[query["bank_id"]], key=lambda row: (row.distance, row.source_bank_id))
        axis_ranked = [row for row in ranked if row.source_bank_id in {item["bank_id"] for item in axis_sources}]
        distances = {row.source_bank_id: row.distance for row in ranked}
        bracket_mean = float(np.mean([distances[item] for item in bracket]))
        outside = [distances[row["bank_id"]] for row in axis_sources if row["bank_id"] not in bracket]
        factor_gap = [abs(np.log(float(row["factor_value"])) - np.log(q)) for row in axis_sources]
        axis_distance = [distances[row["bank_id"]] for row in axis_sources]
        factor_rank, distance_rank = _rank(factor_gap), _rank(axis_distance)
        order = (
            None
            if len(axis_sources) < 3 or np.std(factor_rank) == 0 or np.std(distance_rank) == 0
            else float(np.corrcoef(factor_rank, distance_rank)[0, 1])
        )
        body = {
            "query_bank_id": query["bank_id"],
            "task_id": query["task_id"],
            "axis_id": query["axis_id"],
            "factor_id": query["factor_id"],
            "factor_value": q,
            "bracket_source_ids": sorted(bracket),
            "nearest_is_bracket": float(ranked[0].source_bank_id in bracket),
            "global_top2_are_bracket": float({row.source_bank_id for row in ranked[:2]} == bracket),
            "axis_top2_are_bracket": float({row.source_bank_id for row in axis_ranked[:2]} == bracket),
            "bracket_vs_outside_margin": None if not outside else float(np.mean(outside) - bracket_mean),
            "log_factor_distance_spearman": order,
            "expected_source_diagnostic": query["expected_source_bank_id"],
        }
        details.append({**body, "row_id": sha256_json(body)})
    metrics: dict[str, float] = {
        "interpolation_query_count": float(len(query_rows)),
        "interpolation_bracket_eligible_count": float(len(details)),
        "interpolation_bracket_coverage": float(len(details) / len(query_rows)),
    }
    for field, name in (
        ("nearest_is_bracket", "interpolation_nearest_bracket_rate"),
        ("global_top2_are_bracket", "interpolation_global_bracket_top2_rate"),
        ("axis_top2_are_bracket", "interpolation_axis_bracket_top2_rate"),
        ("bracket_vs_outside_margin", "interpolation_bracket_outside_margin"),
        ("log_factor_distance_spearman", "interpolation_log_factor_spearman"),
    ):
        values = [float(row[field]) for row in details if row[field] is not None]
        if values:
            metrics[name] = float(np.mean(values))
    return details, metrics


def _run_work(
    inputs: Inputs,
    output: Path,
    cell: SignalCell,
    seed: int | None,
    optimization: CorroOptimizationConfig,
    execution: SignalExecutionProtocol,
    resume: bool,
) -> Mapping[str, Any]:
    path = _work_path(output, cell, seed)
    if path.exists():
        if not resume:
            raise SignalBankRunnerError(f"work exists; use --resume: {path}")
        value = read_json(path)
        if value.get("index_digest") != inputs.index_digest or value.get(
            "execution_protocol_digest"
        ) != execution.protocol_digest:
            raise SignalBankRunnerError(f"resume binding changed: {path}")
        return value
    feature_banks = _features(inputs, cell)
    if cell.representation_id in {R2_SOURCE_PCA_WHITEN, R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR}:
        train = _source_split(inputs, cell.condition_id, False)
        source = RepresentationBatch(train.flattened_values(), train.split_digest, "SOURCE_FIT")
    else:
        source = RepresentationBatch(
            np.concatenate([item.values for item in feature_banks[: inputs.source_count]]),
            sha256_json([item.feature_bank_digest for item in feature_banks[: inputs.source_count]]),
            "SOURCE_FIT",
        )
    fitted = _representation(inputs, output, cell, seed, source, optimization)
    represented = transform_feature_banks(fitted, feature_banks)
    sources = represented[: inputs.source_count]
    queries = represented[inputs.source_count : inputs.source_count + inputs.query_count]
    repeats = represented[-inputs.repeat_count :]
    main = run_signal_cell(
        plan=build_signal_matrix_plan(),
        cell=cell,
        source_banks=sources,
        query_banks=queries,
        expected_source_by_query={row["bank_id"]: row["expected_source_bank_id"] for row in inputs.query_rows},
        identity_registry=inputs.registry,
        execution_protocol=execution,
    )
    repeat = run_signal_cell(
        plan=build_signal_matrix_plan(),
        cell=cell,
        source_banks=sources,
        query_banks=repeats,
        expected_source_by_query={
            f"{row['bank_id']}-repeat": row["bank_id"] for row in inputs.source_rows
        },
        identity_registry=inputs.registry,
        execution_protocol=execution,
    )
    interpolation_rows, interpolation_metrics = _interpolation_metrics(
        main.metric_record, inputs.source_rows, inputs.query_rows
    )
    body = {
        "scope": SCOPE,
        "index_digest": inputs.index_digest,
        "execution_protocol_digest": execution.protocol_digest,
        "cell_id": cell.cell_id,
        "condition_id": cell.condition_id,
        "representation_id": cell.representation_id,
        "seed": seed,
        "run": main.to_dict(),
        "kernel_protocol": main.kernel_protocol.to_dict(),
        "metric_record": main.metric_record.to_dict(),
        "diagnostics": main.diagnostics.to_private_dict(),
        "interpolation_metrics": interpolation_metrics,
        "interpolation_rows": interpolation_rows,
        "exact_repeat_run_digest": repeat.run_digest,
        "exact_repeat_rows": _repeat_rows(repeat, inputs.source_count),
    }
    value = {**body, "row_id": sha256_json(body)}
    atomic_write_json(path, value)
    return value


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    data = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, data, overwrite=True)


def _atlas_indexes(output: Path, cells: Sequence[SignalCell], suffix: str) -> Mapping[str, int]:
    selected = {cell.cell_id for cell in cells}
    works, metrics, distances, repeats, interpolation = [], [], [], [], []
    for path in sorted((output / "atlas" / "work").glob("work-*.json")):
        value = read_json(path)
        if value["cell_id"] not in selected:
            continue
        works.append({name: value[name] for name in ("row_id", "cell_id", "condition_id", "representation_id", "seed")} | {"path": str(path)})
        for metric_id, number in value["metric_record"]["metric_values"].items():
            metrics.append({"row_id": sha256_json([value["row_id"], metric_id]), "cell_id": value["cell_id"], "condition_id": value["condition_id"], "representation_id": value["representation_id"], "seed": value["seed"], "metric_family": "signal_runtime", "metric_id": metric_id, "value": number})
        for metric_id, number in value["interpolation_metrics"].items():
            metrics.append({"row_id": sha256_json([value["row_id"], metric_id]), "cell_id": value["cell_id"], "condition_id": value["condition_id"], "representation_id": value["representation_id"], "seed": value["seed"], "metric_family": "interpolation", "metric_id": metric_id, "value": number})
        for row in value["metric_record"]["rows"]:
            distances.append({"row_id": sha256_json([value["row_id"], row["query_bank_id"], row["source_bank_id"]]), "cell_id": value["cell_id"], "condition_id": value["condition_id"], "representation_id": value["representation_id"], "seed": value["seed"], **row})
        repeats.extend(value["exact_repeat_rows"])
        interpolation.extend({"cell_id": value["cell_id"], "condition_id": value["condition_id"], "representation_id": value["representation_id"], "seed": value["seed"], **row} for row in value["interpolation_rows"])
    for name, rows in (("representations", works), ("metrics", metrics), ("distances", distances), ("exact-repeat", repeats), ("interpolation", interpolation)):
        _jsonl(output / f"{name}-{suffix}.jsonl", rows)
    return {"work_rows": len(works), "metric_rows": len(metrics), "distance_rows": len(distances), "exact_repeat_rows": len(repeats), "interpolation_rows": len(interpolation)}


def _run_fits(args, inputs: Inputs, output: Path, optimization: CorroOptimizationConfig):
    jobs = _assigned(_jobs(), args.shard_index, args.shard_count)
    jobs = jobs if args.max_jobs is None else jobs[: args.max_jobs]
    rows, errors = [], []
    for job in jobs:
        try:
            rows.append(_fit_job(inputs, output, job, optimization, args.resume))
        except Exception as error:
            errors.append({"work_id": job.job_id, "error_type": type(error).__name__, "error": str(error)})
    suffix = f"shard-{args.shard_index:05d}-of-{args.shard_count:05d}"
    _jsonl(output / f"fits-{suffix}.jsonl", rows)
    _jsonl(output / f"errors-fits-{suffix}.jsonl", errors)
    return {"scheduled": len(jobs), "complete": len(rows), "failed": len(errors)}


def _run_atlas(args, inputs: Inputs, output: Path, optimization: CorroOptimizationConfig):
    execution = _protocol(inputs, args, optimization)
    cells = _assigned(_cells(), args.shard_index, args.shard_count)
    cells = cells if args.max_cells is None else cells[: args.max_cells]
    complete, errors = [], []
    for cell in cells:
        records = []
        for seed in execution.expected_evaluation_seeds(cell):
            try:
                records.append(_run_work(inputs, output, cell, seed, optimization, execution, args.resume))
            except Exception as error:
                errors.append({"work_id": f"{cell.cell_id}::seed-{seed}", "error_type": type(error).__name__, "error": str(error)})
        if len(records) == len(execution.expected_evaluation_seeds(cell)):
            path = output / "atlas" / "cells" / f"cell-{cell.cell_digest}.json"
            if not path.exists():
                atomic_write_json(path, {"cell_id": cell.cell_id, "cell_digest": cell.cell_digest, "work_rows": [row["row_id"] for row in records], "interpolation_metrics_by_seed": [{"seed": row["seed"], **row["interpolation_metrics"]} for row in records], "status": "COMPLETE"})
            complete.append(cell.cell_id)
    suffix = f"shard-{args.shard_index:05d}-of-{args.shard_count:05d}"
    _jsonl(output / f"errors-atlas-{suffix}.jsonl", errors)
    return {"scheduled": len(cells), "complete": len(complete), "failed": len(errors), "execution_protocol_digest": execution.protocol_digest, **_atlas_indexes(output, cells, suffix)}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.shard_index >= args.shard_count:
        raise SignalBankRunnerError("shard-index must be smaller than shard-count")
    if args.stage == "all" and args.shard_count != 1:
        raise SignalBankRunnerError("parallel execution uses separate fits and atlas stages")
    if not 0 < args.train_fraction < 1 or not 0 < args.validation_fraction < 1 or args.train_fraction + args.validation_fraction >= 1:
        raise SignalBankRunnerError("invalid source episode split")
    output = Path(args.output_dir).expanduser().resolve()
    (output / "atlas" / "work").mkdir(parents=True, exist_ok=True)
    (output / "atlas" / "cells").mkdir(parents=True, exist_ok=True)
    inputs = prepare_signal_inputs(
        Path(args.context_index).expanduser().resolve(),
        args.train_fraction,
        args.validation_fraction,
        args.split_seed,
        args.measurement_episodes,
        args.measurement_transitions,
    )
    optimization = CorroOptimizationConfig(train_steps=args.train_steps)
    summary: dict[str, Any] = {
        "scope": SCOPE,
        "stage": args.stage,
        "index_digest": inputs.index_digest,
        "output_dir": str(output),
        "source_contexts": inputs.source_count,
        "development_queries": inputs.query_count,
        "exact_repeat_pairs": inputs.repeat_count,
        "task_classes": len({row["task_id"] for row in inputs.source_rows}),
        "canonicalizer_digest": inputs.canonicalizer.canonicalizer_digest,
        "source_episode_split": [args.train_fraction, args.validation_fraction, 1 - args.train_fraction - args.validation_fraction],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "train_steps": args.train_steps,
        "measurement": dict(inputs.measurement),
    }
    if args.stage in {"fits", "all"}:
        summary["fits"] = _run_fits(args, inputs, output, optimization)
    if args.stage in {"atlas", "all"}:
        summary["atlas"] = _run_atlas(args, inputs, output, optimization)
    summary["status"] = "COMPLETE_WITH_ERRORS" if any(section.get("failed", 0) for section in (summary.get("fits", {}), summary.get("atlas", {}))) else "COMPLETE"
    atomic_write_json(output / f"summary-{args.stage}-shard-{args.shard_index:05d}-of-{args.shard_count:05d}.json", summary, overwrite=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v0.3 dynamics Signal cells from a flat context-bank index")
    parser.add_argument("--context-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("fits", "atlas", "all"), default="all")
    parser.add_argument("--shard-index", type=_nonnegative, default=0)
    parser.add_argument("--shard-count", type=_positive, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=_nonnegative, default=0)
    parser.add_argument("--measurement-episodes", type=_positive, default=16)
    parser.add_argument("--measurement-transitions", type=_positive, default=64)
    parser.add_argument("--train-steps", type=_nonnegative, default=20_000)
    parser.add_argument("--max-jobs", type=_positive)
    parser.add_argument("--max-cells", type=_positive)
    parser.add_argument("--support-budget", type=_positive, default=100)
    parser.add_argument("--support-steps", type=_nonnegative, default=1_000)
    parser.add_argument("--kmeans-steps", type=_nonnegative, default=25)
    parser.add_argument("--reducer-backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--pair-budget", type=_positive, default=10_000)
    parser.add_argument("--bandwidth-seed", type=_nonnegative, default=0)
    parser.add_argument("--block-size", type=_positive, default=2048)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run(_parser().parse_args(argv))
    except Exception as error:
        raise SystemExit(f"signal-bank runner failed: {type(error).__name__}: {error}") from error
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Inputs", "main", "prepare_signal_inputs", "run"]
