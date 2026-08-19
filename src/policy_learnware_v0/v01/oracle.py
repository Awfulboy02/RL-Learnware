"""Private frozen-policy oracle for the v0.1 diagnostic benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_file, sha256_json
from ..policy.bundle import PolicyBundleMetadata, validate_bundle
from ..policy.evaluate import (
    evaluate_frozen_policy_returns_batched,
    verify_compiled_policy_parity,
)
from ..policy.loader import load_policy
from ..policy.parity import verify_golden_parity
from .schemas import OracleAggregateRecord, OracleEpisodeRecord
from .statistics import mean_bootstrap, paired_transfer_bootstrap


class OracleEvaluationError(RuntimeError):
    """A candidate or environment violated the frozen oracle contract."""


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    task_private: str
    algorithm: str
    training_seed: int
    job_id: str
    bundle_path: str
    bundle_digest: str
    observation_dim: int
    action_dim: int
    outer_iteration: int
    environment_steps: int

    @classmethod
    def from_inventory_item(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        job_id = str(value["job_id"])
        return cls(
            candidate_id=job_id,
            task_private=str(value["task"]),
            algorithm=str(value["algorithm"]),
            training_seed=int(value["training_seed"]),
            job_id=job_id,
            bundle_path=str(value["bundle_dir"]),
            bundle_digest=str(value["bundle_digest"]),
            observation_dim=int(value["observation_dim"]),
            action_dim=int(value["action_dim"]),
            outer_iteration=int(value["outer_iteration"]),
            environment_steps=int(value["environment_steps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedCandidate:
    record: CandidateRecord
    metadata: PolicyBundleMetadata
    policy: Any
    golden_parity: Mapping[str, Any]
    compiled_parity: Mapping[str, Any]


@dataclass(frozen=True)
class OracleShard:
    task_private: str
    variant_id: str
    candidate_id: str
    instance_digest: str
    bundle_digest: str
    evaluator_contract_digest: str
    episodes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-oracle-shard.v0",
            "task_private": self.task_private,
            "variant_id": self.variant_id,
            "candidate_id": self.candidate_id,
            "instance_digest": self.instance_digest,
            "bundle_digest": self.bundle_digest,
            "evaluator_contract_digest": self.evaluator_contract_digest,
            "episodes": list(self.episodes),
        }


def load_candidates_from_inventory(
    inventory_path: str | Path,
    *,
    tasks: Sequence[str],
    candidates_per_task: int,
    checkpoint_outer: int,
    expected_environment_steps: int,
) -> tuple[CandidateRecord, ...]:
    payload = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "policy-learnware.policy-inventory.v0":
        raise OracleEvaluationError("unsupported base policy inventory")
    if int(payload.get("checkpoint_outer", -1)) != int(checkpoint_outer):
        raise OracleEvaluationError("base inventory checkpoint differs from v0.1")
    if int(payload.get("actual_environment_steps", -1)) != int(
        expected_environment_steps
    ):
        raise OracleEvaluationError("base inventory environment-step budget differs")
    allowed_tasks = tuple(tasks)
    records = tuple(
        CandidateRecord.from_inventory_item(item)
        for item in payload.get("items", [])
        if str(item.get("task")) in allowed_tasks
    )
    for task in allowed_tasks:
        selected = [record for record in records if record.task_private == task]
        if len(selected) != int(candidates_per_task):
            raise OracleEvaluationError(
                f"{task} has {len(selected)} candidates, expected {candidates_per_task}"
            )
        algorithm_counts = {
            name: sum(record.algorithm == name for record in selected)
            for name in ("fpo", "ppo")
        }
        if algorithm_counts != {"fpo": 5, "ppo": 5}:
            raise OracleEvaluationError(
                f"{task} candidates are not the frozen 5-FPO/5-PPO pool"
            )
        if {
            (record.algorithm, record.training_seed) for record in selected
        } != {
            (algorithm, seed)
            for algorithm in ("fpo", "ppo")
            for seed in range(5)
        }:
            raise OracleEvaluationError(
                f"{task} inventory does not contain exactly seeds 0..4 per algorithm"
            )
        if any(record.outer_iteration != checkpoint_outer for record in selected):
            raise OracleEvaluationError(f"{task} inventory contains another checkpoint")
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.task_private,
                item.algorithm,
                item.training_seed,
                item.bundle_digest,
            ),
        )
    )


def resolve_candidate_bundle(
    record: CandidateRecord,
    *,
    runs_root: str | Path | None = None,
) -> PolicyBundleMetadata:
    """Relocate a server bundle by job id, then revalidate its exact digest."""

    original = Path(record.bundle_path)
    candidates = [original]
    if runs_root is not None:
        job_root = Path(runs_root)
        if (job_root / "full").is_dir():
            job_root = job_root / "full"
        job_dir = job_root / record.job_id
        candidates.extend(
            sorted(job_dir.glob("attempt_*/checkpoints/outer_000006"), reverse=True)
        )
    for path in candidates:
        if not path.is_dir():
            continue
        metadata = validate_bundle(
            path,
            expected_task=record.task_private,
            expected_algorithm=record.algorithm,
            expected_seed=record.training_seed,
            expected_outer=record.outer_iteration,
            expected_environment_steps=record.environment_steps,
        )
        if metadata.bundle_digest != record.bundle_digest:
            continue
        return metadata
    raise OracleEvaluationError(
        f"cannot resolve a digest-matching bundle for {record.candidate_id}"
    )


def prepare_candidate(
    record: CandidateRecord,
    *,
    fpo_root: str | Path,
    runs_root: str | Path | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> PreparedCandidate:
    metadata = resolve_candidate_bundle(record, runs_root=runs_root)
    policy = load_policy(metadata, fpo_root=fpo_root)
    golden = verify_golden_parity(policy, metadata, atol=atol, rtol=rtol)
    if not golden.passed:
        raise OracleEvaluationError(f"golden parity failed for {record.candidate_id}")
    with np.load(metadata.bundle_dir / "golden_io.npz", allow_pickle=False) as archive:
        observation = np.asarray(archive["observation"], dtype=np.float32)
        key_data = np.asarray(archive["prng_key_data"], dtype=np.uint32)
    compiled = verify_compiled_policy_parity(
        policy,
        observation,
        key_data,
        atol=atol,
        rtol=rtol,
    )
    if not compiled.passed:
        raise OracleEvaluationError(f"compiled parity failed for {record.candidate_id}")
    return PreparedCandidate(
        record=record,
        metadata=metadata,
        policy=policy,
        golden_parity=asdict(golden),
        compiled_parity=asdict(compiled),
    )


def evaluator_contract_digest(*, horizon: int) -> str:
    from ..policy import evaluate as evaluator_module

    return sha256_json(
        {
            "schema": "policy-learnware.v01-oracle-evaluator-contract.v0",
            "horizon": int(horizon),
            "discount": 1.0,
            "primary": "undiscounted_per_step_mean",
            "paired_across_variants": True,
            "paired_across_candidates": False,
            "source_sha256": sha256_file(Path(evaluator_module.__file__)),
        }
    )


def evaluate_oracle_shard(
    prepared: PreparedCandidate,
    adapter: Any,
    *,
    task_private: str,
    variant_id: str,
    instance_digest: str,
    reset_seeds: Sequence[int],
    policy_seeds: Sequence[int],
    horizon: int,
) -> OracleShard:
    if prepared.record.task_private != task_private:
        raise OracleEvaluationError("candidate task differs from private environment task")
    if prepared.record.observation_dim != int(adapter.schema.observation_dim):
        raise OracleEvaluationError("candidate observation schema is incompatible")
    if prepared.record.action_dim != int(adapter.schema.action_dim):
        raise OracleEvaluationError("candidate action schema is incompatible")
    reset_values = tuple(int(value) for value in reset_seeds)
    policy_values = tuple(int(value) for value in policy_seeds)
    if len(reset_values) != len(policy_values) or not reset_values:
        raise OracleEvaluationError("oracle seed vectors are empty or misaligned")
    returns = evaluate_frozen_policy_returns_batched(
        prepared.policy,
        adapter.environment,
        reset_seeds=reset_values,
        policy_seeds=policy_values,
        horizon=int(horizon),
        observation_dim=int(adapter.schema.observation_dim),
        action_dim=int(adapter.schema.action_dim),
    )
    if len(returns) != len(reset_values) or not np.all(np.isfinite(returns)):
        raise OracleEvaluationError("oracle evaluator returned invalid episode results")
    contract_digest = evaluator_contract_digest(horizon=int(horizon))
    episode_rows = tuple(
        {
            "task_private": task_private,
            "variant_id": variant_id,
            "candidate_id": prepared.record.candidate_id,
            "episode_index": index,
            "reset_seed": reset_values[index],
            "policy_seed": policy_values[index],
            "raw_episodic_sum": float(raw_return),
            "mean_step_return": float(raw_return) / float(horizon),
            "instance_digest": instance_digest,
            "bundle_digest": prepared.metadata.bundle_digest,
            "evaluator_contract_digest": contract_digest,
        }
        for index, raw_return in enumerate(returns)
    )
    return OracleShard(
        task_private=task_private,
        variant_id=variant_id,
        candidate_id=prepared.record.candidate_id,
        instance_digest=instance_digest,
        bundle_digest=prepared.metadata.bundle_digest,
        evaluator_contract_digest=contract_digest,
        episodes=episode_rows,
    )


def validate_oracle_shard_payload(
    payload: Mapping[str, Any],
    prepared: PreparedCandidate,
    adapter: Any,
    *,
    task_private: str,
    variant_id: str,
    instance_digest: str,
    reset_seeds: Sequence[int],
    policy_seeds: Sequence[int],
    horizon: int,
) -> OracleShard:
    """Strictly revalidate a persisted shard without rerunning its rollouts."""

    expected_fields = {
        "schema", "task_private", "variant_id", "candidate_id", "instance_digest",
        "bundle_digest", "evaluator_contract_digest", "episodes",
    }
    if set(payload) != expected_fields:
        raise OracleEvaluationError("oracle shard has missing or unknown fields")
    if payload["schema"] != "policy-learnware.v01-oracle-shard.v0":
        raise OracleEvaluationError("unsupported oracle shard schema")
    expected_contract = evaluator_contract_digest(horizon=int(horizon))
    envelope = {
        "task_private": task_private,
        "variant_id": variant_id,
        "candidate_id": prepared.record.candidate_id,
        "instance_digest": instance_digest,
        "bundle_digest": prepared.metadata.bundle_digest,
        "evaluator_contract_digest": expected_contract,
    }
    if any(payload.get(name) != value for name, value in envelope.items()):
        raise OracleEvaluationError("oracle shard envelope differs from frozen inputs")
    if prepared.record.observation_dim != int(adapter.schema.observation_dim):
        raise OracleEvaluationError("resumed candidate observation schema is incompatible")
    if prepared.record.action_dim != int(adapter.schema.action_dim):
        raise OracleEvaluationError("resumed candidate action schema is incompatible")
    reset_values = tuple(int(value) for value in reset_seeds)
    policy_values = tuple(int(value) for value in policy_seeds)
    raw_rows = payload["episodes"]
    if not isinstance(raw_rows, list) or len(raw_rows) != len(reset_values):
        raise OracleEvaluationError("oracle shard episode coverage differs from protocol")
    episode_fields = set(OracleEpisodeRecord.__dataclass_fields__) - {"schema"}
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != episode_fields:
            raise OracleEvaluationError("oracle episode has missing or unknown fields")
        record = OracleEpisodeRecord(**{name: raw[name] for name in episode_fields})
        if (
            record.task_private != task_private
            or record.variant_id != variant_id
            or record.candidate_id != prepared.record.candidate_id
            or record.episode_index != index
            or record.reset_seed != reset_values[index]
            or record.policy_seed != policy_values[index]
            or record.instance_digest != instance_digest
            or record.bundle_digest != prepared.metadata.bundle_digest
            or record.evaluator_contract_digest != expected_contract
        ):
            raise OracleEvaluationError("oracle episode differs from frozen resume inputs")
        rows.append(dict(raw))
    return OracleShard(
        task_private=task_private,
        variant_id=variant_id,
        candidate_id=prepared.record.candidate_id,
        instance_digest=instance_digest,
        bundle_digest=prepared.metadata.bundle_digest,
        evaluator_contract_digest=expected_contract,
        episodes=tuple(rows),
    )


def paired_episode_effects(
    shifted_rows: Sequence[Mapping[str, Any]],
    nominal_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return paired episode deltas after strict seed/index alignment."""

    shifted = sorted(shifted_rows, key=lambda row: int(row["episode_index"]))
    nominal = sorted(nominal_rows, key=lambda row: int(row["episode_index"]))
    if len(shifted) != len(nominal) or not shifted:
        raise OracleEvaluationError("paired oracle rows are empty or misaligned")
    deltas: list[float] = []
    for left, right in zip(shifted, nominal, strict=True):
        for field in ("candidate_id", "episode_index", "reset_seed", "policy_seed"):
            if left[field] != right[field]:
                raise OracleEvaluationError(f"paired oracle rows differ in {field}")
        deltas.append(float(left["mean_step_return"]) - float(right["mean_step_return"]))
    return np.asarray(deltas, dtype=np.float64)


def aggregate_effect_point(deltas: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean(D), abs(mean(D)))`` -- never ``mean(abs(D))``."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired deltas must be a finite non-empty vector")
    delta = float(np.mean(values))
    return delta, abs(delta)


def aggregate_oracle_pair(
    shifted_rows: Sequence[Mapping[str, Any]],
    nominal_rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    mean_seed: int,
    transfer_seed: int,
    confidence_level: float = 0.95,
) -> OracleAggregateRecord:
    """Construct one typed aggregate directly from paired raw episode rows."""

    deltas = paired_episode_effects(shifted_rows, nominal_rows)
    shifted = sorted(shifted_rows, key=lambda row: int(row["episode_index"]))
    shifted_returns = np.asarray(
        [float(row["mean_step_return"]) for row in shifted], dtype=np.float64
    )
    nominal_sorted = sorted(nominal_rows, key=lambda row: int(row["episode_index"]))
    nominal_returns = np.asarray(
        [float(row["mean_step_return"]) for row in nominal_sorted], dtype=np.float64
    )
    mean_result = mean_bootstrap(
        shifted_returns,
        resamples=int(resamples),
        seed=int(mean_seed),
        confidence_level=float(confidence_level),
    )
    transfer = paired_transfer_bootstrap(
        shifted_returns,
        nominal_returns,
        resamples=int(resamples),
        seed=int(transfer_seed),
        confidence_level=float(confidence_level),
    )
    # The explicit point calculation is retained as a guard against a future
    # bootstrap API accidentally redefining J as mean(abs(D)).
    delta, gap = aggregate_effect_point(deltas)
    if not np.isclose(delta, transfer.delta.observed, rtol=0.0, atol=1e-15):
        raise OracleEvaluationError("paired transfer aggregate is internally inconsistent")
    mean_ci = mean_result.interval
    delta_ci = transfer.delta.interval
    gap_ci = transfer.abs_gap_interval
    first = shifted[0]
    nominal_first = nominal_sorted[0]
    return OracleAggregateRecord(
        task_private=str(first["task_private"]),
        variant_id=str(first["variant_id"]),
        nominal_variant_id=str(nominal_first["variant_id"]),
        candidate_id=str(first["candidate_id"]),
        episode_count=len(shifted),
        mean_step_return=mean_result.observed,
        mean_return_ci_low=mean_ci.low,
        mean_return_ci_high=mean_ci.high,
        delta_return=delta,
        delta_ci_low=delta_ci.low,
        delta_ci_high=delta_ci.high,
        abs_transfer_gap=gap,
        abs_gap_ci_low=gap_ci.low,
        abs_gap_ci_high=gap_ci.high,
    )


__all__ = [
    "CandidateRecord",
    "OracleEvaluationError",
    "OracleShard",
    "PreparedCandidate",
    "aggregate_effect_point",
    "aggregate_oracle_pair",
    "evaluate_oracle_shard",
    "evaluator_contract_digest",
    "load_candidates_from_inventory",
    "paired_episode_effects",
    "prepare_candidate",
    "resolve_candidate_bundle",
    "validate_oracle_shard_payload",
]
