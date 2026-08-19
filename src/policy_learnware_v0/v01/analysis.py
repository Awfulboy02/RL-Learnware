"""Pure assembly of the v0.1 OracleMatrix and scientific diagnostics.

This module is the only place where the private context map, the private
oracle, and the selector-safe TaskSpec matrix are joined.  It deliberately
accepts raw episode shards and primitive TaskSpec rows; no externally supplied
``passed`` value, aggregate, p-value, competence set, or ranking decision is
used.

Two facts needed by Gate B are intentionally absent from the public
TaskSpecMatrix and therefore have to be supplied as frozen join evidence:

* ``source_task_by_id`` resolves an opaque source learnware id to its private
  source-task label, which is required to score routing correctness;
* ``schema_view_digest_by_variant`` binds each opaque variant to its frozen
  measurement schema view.  Exact within-task digest identity proves that the
  schema/mask-only negative-control distance is zero.  A mismatch is rejected
  because the current artifacts contain no numeric mask-only embedding from
  which another distance could be reconstructed.

Both inputs are validated fail-closed.  They are private analysis inputs and
must never be copied into measurement artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .config import V01ExperimentConfig
from .gates import (
    CorrectedEffect,
    GateAReport,
    GateBReport,
    GateCDiagnostic,
    RankingReversalEvidence,
    Top1ChangeEvidence,
    evaluate_gate_a,
    evaluate_gate_a_task,
    evaluate_gate_b,
    evaluate_gate_b_task,
    evaluate_gate_c,
    evaluate_gate_d,
)
from .oracle import (
    CandidateRecord,
    aggregate_oracle_pair,
    paired_episode_effects,
)
from .schemas import (
    OracleAggregateRecord,
    OracleEpisodeRecord,
    PrivateContextRecord,
    ShiftManifest,
)
from .statistics import (
    derive_bootstrap_seed,
    holm_bonferroni,
    independent_mean_difference_bootstrap,
    independent_sensitivity_difference_bootstrap,
    paired_transfer_bootstrap,
    top1_bootstrap_probabilities,
)


ORACLE_SHARD_SCHEMA = "policy-learnware.v01-oracle-shard.v0"
TASKSPEC_MATRIX_SCHEMA = "policy-learnware.v01-taskspec-matrix.v0"
SCIENTIFIC_ANALYSIS_SCHEMA = "policy-learnware.v01-scientific-analysis.v0"
_VARIANT_ID = re.compile(r"^v01v-[0-9a-f]{20}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_EXTERNAL_JOIN_EVIDENCE = (
    "source_task_by_id: opaque source learnware id -> private source task",
    "schema_view_digest_by_variant: opaque variant id -> frozen schema-view digest",
)


class AnalysisContractError(ValueError):
    """A raw artifact set is incomplete, inconsistent, or not reconstructible."""


@dataclass(frozen=True)
class ContextBinding:
    task: str
    factor: float
    d_theta: float
    variant_id: str
    private_context_id: str

    @property
    def nominal(self) -> bool:
        return self.factor == 1.0


@dataclass(frozen=True)
class OracleMatrices:
    """Validated raw rows and aggregates reconstructed from those rows."""

    episode_rows: tuple[dict[str, Any], ...]
    aggregates: tuple[OracleAggregateRecord, ...]

    def episodes_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-oracle-episode-matrix.v0",
            "episode_count": len(self.episode_rows),
            "rows": list(self.episode_rows),
        }

    def aggregates_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-oracle-aggregate-matrix.v0",
            "aggregate_count": len(self.aggregates),
            "rows": [record.to_dict() for record in self.aggregates],
        }


@dataclass(frozen=True)
class ScientificAnalysisResult:
    oracle: OracleMatrices
    gate_a: GateAReport
    gate_b: GateBReport
    gate_c: tuple[GateCDiagnostic, ...]
    gate_d_dependency: Mapping[str, Any]
    join_audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCIENTIFIC_ANALYSIS_SCHEMA,
            "oracle_episodes": self.oracle.episodes_dict(),
            "oracle_aggregates": self.oracle.aggregates_dict(),
            "gate_a": self.gate_a.to_dict(),
            "gate_b": self.gate_b.to_dict(),
            "gate_c_diagnostics": [record.to_dict() for record in self.gate_c],
            "gate_d_dependency": dict(self.gate_d_dependency),
            "join_audit": dict(self.join_audit),
        }


def required_join_evidence() -> tuple[str, ...]:
    """Describe the two private primitives not encoded in TaskSpecMatrix."""

    return REQUIRED_EXTERNAL_JOIN_EVIDENCE


def _strict_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise AnalysisContractError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _finite(value: Any, where: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise AnalysisContractError(f"{where} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisContractError(f"{where} must be numeric") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise AnalysisContractError(f"{where} must be {qualifier}")
    return result


def _sha256(value: Any, where: str) -> str:
    result = str(value).lower()
    if _SHA256.fullmatch(result) is None:
        raise AnalysisContractError(f"{where} must be a SHA-256 digest")
    return result


def _variant(value: Any, where: str) -> str:
    result = str(value)
    if _VARIANT_ID.fullmatch(result) is None:
        raise AnalysisContractError(f"{where} is not an opaque v0.1 variant id")
    return result


def _context_entries(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        _strict_keys(value, {"schema", "experiment_id", "entries"}, "private context map")
        if value["schema"] != "policy-learnware.v01-private-context-map.v0":
            raise AnalysisContractError("unsupported private context map schema")
        entries = value["entries"]
    else:
        entries = value
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence) or not entries:
        raise AnalysisContractError("private context entries must be a non-empty sequence")
    if not all(isinstance(item, Mapping) for item in entries):
        raise AnalysisContractError("private context entries must be JSON objects")
    return entries


def parse_context_bindings(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    config: V01ExperimentConfig,
) -> tuple[ContextBinding, ...]:
    """Validate the frozen private context/ShiftManifest join."""

    bindings: list[ContextBinding] = []
    for index, entry in enumerate(_context_entries(value)):
        _strict_keys(
            entry,
            {"context", "shift_manifest", "shift_manifest_digest", "variant_id"},
            f"private context entry {index}",
        )
        try:
            context = PrivateContextRecord.from_dict(entry["context"])
            manifest = ShiftManifest.from_dict(entry["shift_manifest"])
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(f"invalid private context entry {index}: {exc}") from exc
        if manifest.digest != _sha256(
            entry["shift_manifest_digest"], f"context[{index}].shift_manifest_digest"
        ):
            raise AnalysisContractError("ShiftManifest digest differs from frozen context map")
        if (
            manifest.private_context_id != context.private_context_id
            or manifest.task != context.task
            or manifest.shift_id != context.shift_id
            or manifest.factor != context.factor
        ):
            raise AnalysisContractError("context and ShiftManifest bindings disagree")
        bindings.append(
            ContextBinding(
                task=context.task,
                factor=context.factor,
                d_theta=context.d_theta,
                variant_id=_variant(entry["variant_id"], f"context[{index}].variant_id"),
                private_context_id=context.private_context_id,
            )
        )
    if len({item.variant_id for item in bindings}) != len(bindings):
        raise AnalysisContractError("private context map contains duplicate variant ids")
    if len({item.private_context_id for item in bindings}) != len(bindings):
        raise AnalysisContractError("private context map contains duplicate context ids")
    expected = {
        (task, float(factor))
        for task in config.tasks.all
        for factor in config.shift.diagnostic_grid
    }
    observed = {(item.task, item.factor) for item in bindings}
    if observed != expected or len(bindings) != len(expected):
        raise AnalysisContractError("private context coverage differs from frozen task/factor grid")
    for task in config.tasks.all:
        if sum(item.task == task and item.nominal for item in bindings) != 1:
            raise AnalysisContractError(f"{task} does not have exactly one nominal context")
    order = {task: index for index, task in enumerate(config.tasks.all)}
    return tuple(sorted(bindings, key=lambda item: (order[item.task], item.factor)))


def _candidate_values(
    value: Mapping[str, Any] | Sequence[CandidateRecord | Mapping[str, Any]],
) -> Sequence[CandidateRecord | Mapping[str, Any]]:
    if isinstance(value, Mapping):
        _strict_keys(value, {"schema", "oracle_protocol_id", "candidates"}, "candidate manifest")
        if value["schema"] != "policy-learnware.v01-candidates.v0":
            raise AnalysisContractError("unsupported candidate manifest schema")
        _sha256(value["oracle_protocol_id"], "candidate manifest oracle_protocol_id")
        records = value["candidates"]
    else:
        records = value
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise AnalysisContractError("candidate records must be a non-empty sequence")
    return records


def parse_candidate_records(
    value: Mapping[str, Any] | Sequence[CandidateRecord | Mapping[str, Any]],
    *,
    config: V01ExperimentConfig,
) -> tuple[CandidateRecord, ...]:
    records: list[CandidateRecord] = []
    expected_fields = set(CandidateRecord.__dataclass_fields__)
    for index, raw in enumerate(_candidate_values(value)):
        if isinstance(raw, CandidateRecord):
            record = raw
        elif isinstance(raw, Mapping):
            _strict_keys(raw, expected_fields, f"candidate record {index}")
            try:
                record = CandidateRecord(**{key: raw[key] for key in expected_fields})
            except (TypeError, ValueError) as exc:
                raise AnalysisContractError(f"invalid candidate record {index}: {exc}") from exc
        else:
            raise AnalysisContractError("candidate records must be objects")
        if not record.candidate_id or record.candidate_id != record.job_id:
            raise AnalysisContractError("candidate_id must be the non-empty frozen job_id")
        if record.task_private not in config.tasks.all:
            raise AnalysisContractError("candidate belongs to a task outside the frozen scope")
        if record.algorithm not in {"fpo", "ppo"}:
            raise AnalysisContractError("candidate algorithm must be fpo or ppo")
        if record.outer_iteration != config.base.checkpoint_outer:
            raise AnalysisContractError("candidate checkpoint differs from frozen outer iteration")
        if record.environment_steps != config.base.actual_environment_steps:
            raise AnalysisContractError("candidate training budget differs from frozen budget")
        _sha256(record.bundle_digest, f"candidate[{record.candidate_id}].bundle_digest")
        records.append(record)
    if len({record.candidate_id for record in records}) != len(records):
        raise AnalysisContractError("candidate records contain duplicate ids")
    for task in config.tasks.all:
        selected = [record for record in records if record.task_private == task]
        if len(selected) != config.base.candidates_per_task:
            raise AnalysisContractError(
                f"{task} candidate count is {len(selected)}, expected {config.base.candidates_per_task}"
            )
        if config.base.candidates_per_task == 10:
            counts = {name: sum(item.algorithm == name for item in selected) for name in ("fpo", "ppo")}
            if counts != {"fpo": 5, "ppo": 5}:
                raise AnalysisContractError(f"{task} is not the frozen 5-FPO/5-PPO pool")
            observed_algorithm_seeds = {
                (item.algorithm, item.training_seed) for item in selected
            }
            expected_algorithm_seeds = {
                (algorithm, seed)
                for algorithm in ("fpo", "ppo")
                for seed in range(5)
            }
            if observed_algorithm_seeds != expected_algorithm_seeds:
                raise AnalysisContractError(
                    f"{task} candidates do not contain exactly seeds 0..4 per algorithm"
                )
    task_order = {task: index for index, task in enumerate(config.tasks.all)}
    return tuple(
        sorted(records, key=lambda item: (task_order[item.task_private], item.candidate_id))
    )


def _normalise_shards(value: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise AnalysisContractError("oracle shards must be a non-empty sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise AnalysisContractError("oracle shards must be JSON objects")
    return tuple(value)


def _parse_oracle_rows(
    shards: Sequence[Mapping[str, Any]],
    *,
    bindings: Sequence[ContextBinding],
    candidates: Sequence[CandidateRecord],
    config: V01ExperimentConfig,
) -> dict[tuple[str, str, str], tuple[OracleEpisodeRecord, ...]]:
    binding_by_variant = {item.variant_id: item for item in bindings}
    candidate_by_id = {item.candidate_id: item for item in candidates}
    result: dict[tuple[str, str, str], tuple[OracleEpisodeRecord, ...]] = {}
    instance_by_variant: dict[str, str] = {}
    evaluator_digests: set[str] = set()
    seed_vectors: dict[tuple[str, str], tuple[tuple[int, int], ...]] = {}
    episode_fields = set(OracleEpisodeRecord.__dataclass_fields__) - {"schema"}
    shard_fields = {
        "schema", "task_private", "variant_id", "candidate_id", "instance_digest",
        "bundle_digest", "evaluator_contract_digest", "episodes",
    }
    for shard_index, shard in enumerate(_normalise_shards(shards)):
        _strict_keys(shard, shard_fields, f"oracle shard {shard_index}")
        if shard["schema"] != ORACLE_SHARD_SCHEMA:
            raise AnalysisContractError("unsupported oracle shard schema")
        task = str(shard["task_private"])
        variant_id = _variant(shard["variant_id"], f"oracle shard {shard_index} variant_id")
        candidate_id = str(shard["candidate_id"])
        try:
            binding = binding_by_variant[variant_id]
            candidate = candidate_by_id[candidate_id]
        except KeyError as exc:
            raise AnalysisContractError("oracle shard references an unknown variant/candidate") from exc
        if binding.task != task or candidate.task_private != task:
            raise AnalysisContractError("oracle shard crosses task boundaries")
        instance_digest = _sha256(shard["instance_digest"], "oracle shard instance_digest")
        bundle_digest = _sha256(shard["bundle_digest"], "oracle shard bundle_digest")
        evaluator_digest = _sha256(
            shard["evaluator_contract_digest"], "oracle shard evaluator_contract_digest"
        )
        if bundle_digest != candidate.bundle_digest:
            raise AnalysisContractError("oracle shard bundle digest differs from candidate record")
        previous_instance = instance_by_variant.setdefault(variant_id, instance_digest)
        if previous_instance != instance_digest:
            raise AnalysisContractError("one variant is bound to multiple environment instances")
        evaluator_digests.add(evaluator_digest)
        raw_episodes = shard["episodes"]
        if not isinstance(raw_episodes, list) or len(raw_episodes) != config.oracle.episodes_per_candidate_variant:
            raise AnalysisContractError("oracle shard has incorrect episode count")
        episodes: list[OracleEpisodeRecord] = []
        for episode_index, raw in enumerate(raw_episodes):
            if not isinstance(raw, Mapping):
                raise AnalysisContractError("oracle episode is not a JSON object")
            _strict_keys(raw, episode_fields, "oracle episode")
            try:
                episode = OracleEpisodeRecord(**{key: raw[key] for key in episode_fields})
            except (TypeError, ValueError) as exc:
                raise AnalysisContractError(f"invalid oracle episode: {exc}") from exc
            if (
                episode.task_private != task
                or episode.variant_id != variant_id
                or episode.candidate_id != candidate_id
                or episode.instance_digest != instance_digest
                or episode.bundle_digest != bundle_digest
                or episode.evaluator_contract_digest != evaluator_digest
            ):
                raise AnalysisContractError("oracle episode differs from its shard envelope")
            if episode.episode_index != episode_index:
                raise AnalysisContractError("oracle episode indices must be canonical 0..E-1")
            episodes.append(episode)
        key = (task, variant_id, candidate_id)
        if key in result:
            raise AnalysisContractError("duplicate oracle shard work unit")
        result[key] = tuple(episodes)
        vector = tuple((item.reset_seed, item.policy_seed) for item in episodes)
        stream_key = (task, candidate_id)
        previous_vector = seed_vectors.setdefault(stream_key, vector)
        if previous_vector != vector:
            raise AnalysisContractError("oracle episode seeds are not paired across variants")
    if len(evaluator_digests) != 1:
        raise AnalysisContractError("oracle shards use more than one evaluator contract")
    expected = {
        (binding.task, binding.variant_id, candidate.candidate_id)
        for binding in bindings
        for candidate in candidates
        if candidate.task_private == binding.task
    }
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise AnalysisContractError(
            f"oracle shard matrix is incomplete; missing={missing[:3]}, extra={extra[:3]}"
        )
    # Different candidates are explicitly independent.  Reuse across variants
    # has already been collapsed by stream_key above.
    seen_seed_pairs: dict[tuple[int, int], tuple[str, str]] = {}
    for stream_key, vector in seed_vectors.items():
        for pair in vector:
            previous = seen_seed_pairs.setdefault(pair, stream_key)
            if previous != stream_key:
                raise AnalysisContractError("oracle seed collision across candidate streams")
    return result


def _rows_as_mappings(records: Sequence[OracleEpisodeRecord]) -> tuple[dict[str, Any], ...]:
    return tuple(record.to_dict() for record in records)


def build_oracle_matrices(
    private_contexts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate_records: Mapping[str, Any] | Sequence[CandidateRecord | Mapping[str, Any]],
    oracle_shards: Sequence[Mapping[str, Any]],
    *,
    config: V01ExperimentConfig,
    analysis_seed_namespace: str,
) -> tuple[
    tuple[ContextBinding, ...],
    tuple[CandidateRecord, ...],
    Mapping[tuple[str, str, str], tuple[OracleEpisodeRecord, ...]],
    OracleMatrices,
]:
    """Rebuild episode and aggregate matrices from raw immutable shards."""

    if not str(analysis_seed_namespace):
        raise AnalysisContractError("analysis_seed_namespace cannot be empty")
    bindings = parse_context_bindings(private_contexts, config=config)
    candidates = parse_candidate_records(candidate_records, config=config)
    episode_map = _parse_oracle_rows(
        oracle_shards, bindings=bindings, candidates=candidates, config=config
    )
    candidate_ids_by_task = {
        task: tuple(sorted(item.candidate_id for item in candidates if item.task_private == task))
        for task in config.tasks.all
    }
    aggregates: list[OracleAggregateRecord] = []
    ordered_rows: list[dict[str, Any]] = []
    for task in config.tasks.all:
        task_bindings = [item for item in bindings if item.task == task]
        nominal = next(item for item in task_bindings if item.nominal)
        for binding in task_bindings:
            for candidate_id in candidate_ids_by_task[task]:
                shifted = episode_map[(task, binding.variant_id, candidate_id)]
                nominal_rows = episode_map[(task, nominal.variant_id, candidate_id)]
                shifted_dicts = _rows_as_mappings(shifted)
                nominal_dicts = _rows_as_mappings(nominal_rows)
                aggregates.append(
                    aggregate_oracle_pair(
                        shifted_dicts,
                        nominal_dicts,
                        resamples=config.statistics.bootstrap_resamples,
                        mean_seed=derive_bootstrap_seed(
                            analysis_seed_namespace, task, "aggregate_mean", binding.variant_id,
                            candidate_id,
                        ),
                        transfer_seed=derive_bootstrap_seed(
                            analysis_seed_namespace, task, "aggregate_transfer", binding.variant_id,
                            candidate_id,
                        ),
                        confidence_level=config.statistics.confidence_level,
                    )
                )
                ordered_rows.extend(shifted_dicts)
    return bindings, candidates, episode_map, OracleMatrices(tuple(ordered_rows), tuple(aggregates))


def _correct_effect_family(
    records: Sequence[tuple[str, str, tuple[str, ...], Any]],
) -> tuple[CorrectedEffect, ...]:
    """Apply Holm to a complete family of ``(id, context, candidates, bootstrap)``."""

    if not records:
        raise AnalysisContractError("a preregistered effect family is empty")
    holm = holm_bonferroni(
        {hypothesis_id: bootstrap.centered_p_value for hypothesis_id, _, _, bootstrap in records}
    )
    corrected: list[CorrectedEffect] = []
    for hypothesis_id, context_id, candidate_ids, bootstrap in records:
        adjustment = holm[hypothesis_id]
        corrected.append(
            CorrectedEffect(
                hypothesis_id=hypothesis_id,
                context_id=context_id,
                candidate_ids=candidate_ids,
                estimate=bootstrap.observed,
                interval=bootstrap.interval,
                raw_p_value=bootstrap.centered_p_value,
                adjusted_p_value=adjustment.adjusted_p_value,
                correction_order=adjustment.correction_order,
                family_size=adjustment.family_size,
            )
        )
    return tuple(corrected)


def build_gate_a(
    *,
    bindings: Sequence[ContextBinding],
    candidates: Sequence[CandidateRecord],
    episode_map: Mapping[tuple[str, str, str], Sequence[OracleEpisodeRecord]],
    config: V01ExperimentConfig,
    analysis_seed_namespace: str,
) -> GateAReport:
    """Recompute all Gate-A families before competence filtering."""

    task_results = []
    for task in config.tasks.all:
        task_bindings = tuple(item for item in bindings if item.task == task)
        nominal = next(item for item in task_bindings if item.nominal)
        shifted_bindings = tuple(item for item in task_bindings if not item.nominal)
        candidate_ids = tuple(
            sorted(item.candidate_id for item in candidates if item.task_private == task)
        )

        def returns(variant_id: str, candidate_id: str) -> np.ndarray:
            return np.asarray(
                [
                    record.mean_step_return
                    for record in episode_map[(task, variant_id, candidate_id)]
                ],
                dtype=np.float64,
            )

        nominal_returns = {
            candidate_id: float(returns(nominal.variant_id, candidate_id).mean())
            for candidate_id in candidate_ids
        }
        paired_differences: dict[tuple[str, str], np.ndarray] = {}
        material_uncorrected: list[tuple[str, str, tuple[str, ...], Any]] = []
        for binding in shifted_bindings:
            for candidate_id in candidate_ids:
                shifted = returns(binding.variant_id, candidate_id)
                base = returns(nominal.variant_id, candidate_id)
                paired_differences[(binding.variant_id, candidate_id)] = shifted - base
                hypothesis = f"material:{task}:{binding.variant_id}:{candidate_id}"
                material_uncorrected.append(
                    (
                        hypothesis,
                        binding.variant_id,
                        (candidate_id,),
                        paired_transfer_bootstrap(
                            shifted,
                            base,
                            resamples=config.statistics.bootstrap_resamples,
                            seed=derive_bootstrap_seed(
                                analysis_seed_namespace, task, "material", binding.variant_id,
                                candidate_id,
                            ),
                            confidence_level=config.statistics.confidence_level,
                        ).delta,
                    )
                )
        material = _correct_effect_family(material_uncorrected)

        heterogeneity_uncorrected: list[tuple[str, str, tuple[str, ...], Any]] = []
        pairs = tuple(itertools.combinations(candidate_ids, 2))
        for binding in shifted_bindings:
            for left, right in pairs:
                hypothesis = f"heterogeneity:{task}:{binding.variant_id}:{left}:{right}"
                heterogeneity_uncorrected.append(
                    (
                        hypothesis,
                        binding.variant_id,
                        (left, right),
                        independent_sensitivity_difference_bootstrap(
                            paired_differences[(binding.variant_id, left)],
                            paired_differences[(binding.variant_id, right)],
                            resamples=config.statistics.bootstrap_resamples,
                            seed=derive_bootstrap_seed(
                                analysis_seed_namespace, task, "heterogeneity",
                                binding.variant_id, left, right,
                            ),
                            confidence_level=config.statistics.confidence_level,
                        ),
                    )
                )
        heterogeneity = _correct_effect_family(heterogeneity_uncorrected)

        ranking_uncorrected: list[tuple[str, str, tuple[str, ...], Any]] = []
        for left, right in pairs:
            hypothesis = f"ranking:{task}:nominal:{left}:{right}"
            ranking_uncorrected.append(
                (
                    hypothesis,
                    nominal.variant_id,
                    (left, right),
                    independent_mean_difference_bootstrap(
                        returns(nominal.variant_id, left),
                        returns(nominal.variant_id, right),
                        resamples=config.statistics.bootstrap_resamples,
                        seed=derive_bootstrap_seed(
                            analysis_seed_namespace, task, "ranking_nominal", left, right
                        ),
                        confidence_level=config.statistics.confidence_level,
                    ),
                )
            )
        for binding in shifted_bindings:
            for left, right in pairs:
                hypothesis = f"ranking:{task}:{binding.variant_id}:{left}:{right}"
                ranking_uncorrected.append(
                    (
                        hypothesis,
                        binding.variant_id,
                        (left, right),
                        independent_mean_difference_bootstrap(
                            returns(binding.variant_id, left),
                            returns(binding.variant_id, right),
                            resamples=config.statistics.bootstrap_resamples,
                            seed=derive_bootstrap_seed(
                                analysis_seed_namespace, task, "ranking_shifted",
                                binding.variant_id, left, right,
                            ),
                            confidence_level=config.statistics.confidence_level,
                        ),
                    )
                )
        ranking = _correct_effect_family(ranking_uncorrected)
        ranking_by_key = {
            (record.context_id, record.candidate_ids): record for record in ranking
        }
        reversals = tuple(
            RankingReversalEvidence(
                binding.variant_id,
                ranking_by_key[(nominal.variant_id, pair)],
                ranking_by_key[(binding.variant_id, pair)],
            )
            for binding in shifted_bindings
            for pair in pairs
        )

        nominal_top1 = top1_bootstrap_probabilities(
            {candidate_id: returns(nominal.variant_id, candidate_id) for candidate_id in candidate_ids},
            resamples=config.statistics.bootstrap_resamples,
            seed=derive_bootstrap_seed(analysis_seed_namespace, task, "top1", "nominal"),
        )
        top1 = tuple(
            Top1ChangeEvidence(
                binding.variant_id,
                nominal_top1,
                top1_bootstrap_probabilities(
                    {
                        candidate_id: returns(binding.variant_id, candidate_id)
                        for candidate_id in candidate_ids
                    },
                    resamples=config.statistics.bootstrap_resamples,
                    seed=derive_bootstrap_seed(
                        analysis_seed_namespace, task, "top1", binding.variant_id
                    ),
                ),
            )
            for binding in shifted_bindings
        )
        task_results.append(
            evaluate_gate_a_task(
                task=task,
                nominal_returns=nominal_returns,
                context_ids=[item.variant_id for item in shifted_bindings],
                material_effects=material,
                heterogeneity_effects=heterogeneity,
                ranking_reversals=reversals,
                top1_changes=top1,
                competence_alpha=config.statistics.competence_alpha,
                minimum_material_effect=config.statistics.minimum_material_effect,
                minimum_sensitivity_heterogeneity=(
                    config.statistics.minimum_sensitivity_heterogeneity
                ),
                significance_alpha=1.0 - config.statistics.confidence_level,
                top1_bootstrap_probability=config.statistics.top1_bootstrap_probability,
            )
        )
    return evaluate_gate_a(task_results)


def _taskspec_payload(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        if not hasattr(value, "to_dict"):
            raise AnalysisContractError("TaskSpecMatrix must be a mapping or expose to_dict()")
        value = value.to_dict()
    expected = {"schema", "plan_digest", "pair_rows", "routing_rows", "self_norm_rows", "clamp_count"}
    _strict_keys(value, expected, "TaskSpecMatrix")
    if value["schema"] != TASKSPEC_MATRIX_SCHEMA:
        raise AnalysisContractError("unsupported TaskSpecMatrix schema")
    _sha256(value["plan_digest"], "TaskSpecMatrix plan_digest")
    if isinstance(value["clamp_count"], bool) or int(value["clamp_count"]) < 0:
        raise AnalysisContractError("TaskSpecMatrix clamp_count must be non-negative")
    return value


def _validated_taskspec_rows(
    value: Mapping[str, Any] | Any,
    *,
    bindings: Sequence[ContextBinding],
    config: V01ExperimentConfig,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    payload = _taskspec_payload(value)
    pair_rows = payload["pair_rows"]
    routing_rows = payload["routing_rows"]
    if not isinstance(pair_rows, list) or not isinstance(routing_rows, list):
        raise AnalysisContractError("TaskSpecMatrix row collections must be lists")
    binding_by_variant = {item.variant_id: item for item in bindings}
    pair_fields = {
        "family", "pair_index", "left_variant_id", "left_bank", "right_variant_id",
        "right_bank", "prefix", "raw_mmd2", "mmd2", "d_phi", "roundoff_clamped",
        "cross_term",
    }
    seen_pair_keys: set[tuple[Any, ...]] = set()
    observed_within: set[tuple[str, int, int]] = set()
    observed_between: set[tuple[str, int]] = set()
    family_indices: dict[str, set[int]] = {"within": set(), "between": set()}
    for row in pair_rows:
        if not isinstance(row, Mapping):
            raise AnalysisContractError("TaskSpec pair row must be a JSON object")
        _strict_keys(row, pair_fields, "TaskSpec pair row")
        family = str(row["family"])
        if family not in {"within", "between"}:
            raise AnalysisContractError("TaskSpec pair family must be within or between")
        left = _variant(row["left_variant_id"], "TaskSpec left_variant_id")
        right = _variant(row["right_variant_id"], "TaskSpec right_variant_id")
        if left not in binding_by_variant or right not in binding_by_variant:
            raise AnalysisContractError("TaskSpec pair references an unknown variant")
        if binding_by_variant[left].task != binding_by_variant[right].task:
            raise AnalysisContractError("TaskSpec pair crosses task boundaries")
        for name in ("left_bank", "right_bank", "prefix", "pair_index"):
            if isinstance(row[name], bool) or int(row[name]) < 0:
                raise AnalysisContractError(f"TaskSpec {name} must be non-negative")
        family_indices[family].add(int(row["pair_index"]))
        if int(row["prefix"]) != config.probe.gate_b_unreduced_prefix:
            raise AnalysisContractError("TaskSpec Gate-B pair uses an unregistered prefix")
        raw = _finite(row["raw_mmd2"], "raw_mmd2")
        mmd2 = _finite(row["mmd2"], "mmd2", nonnegative=True)
        d_phi = _finite(row["d_phi"], "d_phi", nonnegative=True)
        _finite(row["cross_term"], "cross_term")
        if type(row["roundoff_clamped"]) is not bool:
            raise AnalysisContractError("roundoff_clamped must be boolean")
        if not math.isclose(d_phi * d_phi, mmd2, rel_tol=1e-10, abs_tol=1e-12):
            raise AnalysisContractError("TaskSpec d_phi is inconsistent with mmd2")
        if not math.isclose(mmd2, max(raw, 0.0), rel_tol=1e-10, abs_tol=1e-12):
            raise AnalysisContractError("TaskSpec mmd2 is inconsistent with raw_mmd2")
        if raw < 0.0 and not row["roundoff_clamped"]:
            raise AnalysisContractError("negative raw_mmd2 lacks its clamp attestation")
        key = (family, left, int(row["left_bank"]), right, int(row["right_bank"]))
        if key in seen_pair_keys:
            raise AnalysisContractError("TaskSpec pair rows contain a duplicate primitive")
        seen_pair_keys.add(key)
        if family == "within":
            if left != right or int(row["left_bank"]) == int(row["right_bank"]):
                raise AnalysisContractError("within row must compare two banks of one variant")
            observed_within.add((left, int(row["left_bank"]), int(row["right_bank"])))
        else:
            left_binding = binding_by_variant[left]
            right_binding = binding_by_variant[right]
            if not left_binding.nominal or right_binding.nominal:
                raise AnalysisContractError("between row must be nominal-to-non-nominal")
            if int(row["left_bank"]) != int(row["right_bank"]):
                raise AnalysisContractError("between row must use the same bank")
            observed_between.add((right, int(row["right_bank"])))

    expected_within_keys = {
        (binding.variant_id, int(pair[0]), int(pair[1]))
        for binding in bindings
        for pair in config.probe.sparse_within_bank_pairs
    }
    expected_between_keys = {
        (binding.variant_id, bank)
        for binding in bindings
        if not binding.nominal
        for bank in range(config.probe.banks)
    }
    if observed_within != expected_within_keys:
        raise AnalysisContractError("TaskSpec within-family bank plan differs from frozen plan")
    if observed_between != expected_between_keys:
        raise AnalysisContractError("TaskSpec between-family bank plan differs from frozen plan")
    for family, expected_size in (
        ("within", len(expected_within_keys)),
        ("between", len(expected_between_keys)),
    ):
        if family_indices[family] != set(range(expected_size)):
            raise AnalysisContractError(f"TaskSpec {family} pair indices are not canonical")

    routing_fields = {
        "routing_index", "variant_id", "bank", "prefix", "selected_source_id", "ranking"
    }
    seen_routing: set[tuple[str, int]] = set()
    routing_indices: set[int] = set()
    for row in routing_rows:
        if not isinstance(row, Mapping):
            raise AnalysisContractError("TaskSpec routing row must be a JSON object")
        _strict_keys(row, routing_fields, "TaskSpec routing row")
        variant_id = _variant(row["variant_id"], "TaskSpec routing variant_id")
        if variant_id not in binding_by_variant:
            raise AnalysisContractError("TaskSpec routing references an unknown variant")
        if isinstance(row["routing_index"], bool) or int(row["routing_index"]) < 0:
            raise AnalysisContractError("TaskSpec routing_index must be non-negative")
        routing_indices.add(int(row["routing_index"]))
        if isinstance(row["bank"], bool):
            raise AnalysisContractError("TaskSpec routing bank must be an integer")
        bank = int(row["bank"])
        if bank < 0 or bank >= config.probe.banks:
            raise AnalysisContractError("TaskSpec routing bank is outside the frozen range")
        if int(row["prefix"]) != config.probe.max_episodes_per_bank:
            raise AnalysisContractError("TaskSpec routing uses an unregistered prefix")
        key = (variant_id, bank)
        if key in seen_routing:
            raise AnalysisContractError("TaskSpec routing rows contain a duplicate work unit")
        seen_routing.add(key)
        ranking = row["ranking"]
        if not isinstance(ranking, list) or not ranking:
            raise AnalysisContractError("TaskSpec routing ranking is empty")
        parsed_ranking: list[tuple[str, float]] = []
        for item in ranking:
            if not isinstance(item, Mapping):
                raise AnalysisContractError("TaskSpec routing rank must be an object")
            _strict_keys(item, {"source_id", "routing_score"}, "TaskSpec routing rank")
            source_id = str(item["source_id"])
            if not source_id:
                raise AnalysisContractError("TaskSpec routing source id cannot be empty")
            parsed_ranking.append(
                (source_id, _finite(item["routing_score"], "routing_score"))
            )
        if len({item[0] for item in parsed_ranking}) != len(parsed_ranking):
            raise AnalysisContractError("TaskSpec routing ranking repeats a source")
        expected_order = sorted(parsed_ranking, key=lambda item: (item[1], item[0]))
        if parsed_ranking != expected_order or str(row["selected_source_id"]) != parsed_ranking[0][0]:
            raise AnalysisContractError("TaskSpec routing ranking/selection is inconsistent")
    expected_routing = {
        (binding.variant_id, bank)
        for binding in bindings
        for bank in range(config.probe.banks)
    }
    if seen_routing != expected_routing:
        raise AnalysisContractError("TaskSpec routing matrix is incomplete")
    if routing_indices != set(range(len(expected_routing))):
        raise AnalysisContractError("TaskSpec routing indices are not canonical")
    expected_within = len(bindings) * len(config.probe.sparse_within_bank_pairs)
    expected_between = (
        len(config.tasks.all)
        * (len(config.shift.diagnostic_grid) - 1)
        * config.probe.banks
    )
    if sum(row["family"] == "within" for row in pair_rows) != expected_within:
        raise AnalysisContractError("TaskSpec within-family coverage is incomplete")
    if sum(row["family"] == "between" for row in pair_rows) != expected_between:
        raise AnalysisContractError("TaskSpec between-family coverage is incomplete")
    if int(payload["clamp_count"]) != sum(bool(row["roundoff_clamped"]) for row in pair_rows):
        raise AnalysisContractError("TaskSpec clamp_count differs from primitive rows")
    return tuple(pair_rows), tuple(routing_rows)


def _validate_join_evidence(
    *,
    bindings: Sequence[ContextBinding],
    routing_rows: Sequence[Mapping[str, Any]],
    source_task_by_id: Mapping[str, str],
    schema_view_digest_by_variant: Mapping[str, str],
) -> dict[str, float]:
    if not isinstance(source_task_by_id, Mapping) or not source_task_by_id:
        raise AnalysisContractError(
            "missing source_task_by_id: routing correctness is not reconstructible from opaque rows"
        )
    source_map = {str(key): str(value) for key, value in source_task_by_id.items()}
    if any(not key or not value for key, value in source_map.items()):
        raise AnalysisContractError("source_task_by_id contains an empty id/task")
    ranking_source_sets = [
        {str(item["source_id"]) for item in row["ranking"]} for row in routing_rows
    ]
    if any(source_set != set(source_map) for source_set in ranking_source_sets):
        raise AnalysisContractError("source_task_by_id does not exactly cover every routing ranking")
    if len(set(source_map.values())) != len(source_map):
        raise AnalysisContractError("source_task_by_id must bind one source RKME per task")

    if not isinstance(schema_view_digest_by_variant, Mapping):
        raise AnalysisContractError(
            "missing schema_view_digest_by_variant: mask/schema control is not reconstructible"
        )
    expected_variants = {item.variant_id for item in bindings}
    if set(schema_view_digest_by_variant) != expected_variants:
        raise AnalysisContractError(
            "schema_view_digest_by_variant does not exactly cover the frozen variants"
        )
    digests = {
        _variant(key, "schema-view variant id"): _sha256(value, "schema-view digest")
        for key, value in schema_view_digest_by_variant.items()
    }
    mask_distances: dict[str, float] = {}
    for task in sorted({item.task for item in bindings}):
        task_digests = {digests[item.variant_id] for item in bindings if item.task == task}
        if len(task_digests) != 1:
            raise AnalysisContractError(
                f"{task} schema views differ; numeric mask-only distance is unavailable"
            )
        # The mask-only representation is a deterministic function of the
        # frozen schema view.  Exact digest identity therefore implies an exact
        # zero distance without importing private context into measurement.
        mask_distances[task] = 0.0
    return mask_distances


def build_gate_b(
    taskspec_matrix: Mapping[str, Any] | Any,
    *,
    bindings: Sequence[ContextBinding],
    config: V01ExperimentConfig,
    source_task_by_id: Mapping[str, str],
    schema_view_digest_by_variant: Mapping[str, str],
    gate_d_checks: Mapping[str, bool],
) -> tuple[GateBReport, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Privately join primitive TaskSpec rows and recompute Gate B."""

    pair_rows, routing_rows = _validated_taskspec_rows(
        taskspec_matrix, bindings=bindings, config=config
    )
    mask_distances = _validate_join_evidence(
        bindings=bindings,
        routing_rows=routing_rows,
        source_task_by_id=source_task_by_id,
        schema_view_digest_by_variant=schema_view_digest_by_variant,
    )
    binding_by_variant = {item.variant_id: item for item in bindings}
    task_results = []
    for task in config.tasks.all:
        task_bindings = tuple(item for item in bindings if item.task == task)
        within = [
            float(row["d_phi"])
            for row in pair_rows
            if row["family"] == "within"
            and binding_by_variant[str(row["left_variant_id"])].task == task
        ]
        between_by_variant: dict[str, list[float]] = {
            item.variant_id: [] for item in task_bindings if not item.nominal
        }
        between: list[float] = []
        for row in pair_rows:
            if row["family"] != "between":
                continue
            right = binding_by_variant[str(row["right_variant_id"])]
            if right.task == task:
                distance = float(row["d_phi"])
                between.append(distance)
                between_by_variant[right.variant_id].append(distance)
        severity_bindings = tuple(item for item in task_bindings if not item.nominal)
        if any(len(between_by_variant[item.variant_id]) != config.probe.banks for item in severity_bindings):
            raise AnalysisContractError("TaskSpec severity bank coverage is incomplete")
        task_results.append(
            evaluate_gate_b_task(
                task=task,
                within_distances=within,
                between_distances=between,
                severity_d_theta=[item.d_theta for item in severity_bindings],
                severity_median_d_phi=[
                    float(np.median(between_by_variant[item.variant_id]))
                    for item in severity_bindings
                ],
                mask_schema_max_distance=mask_distances[task],
                minimum_between_within_ratio=(
                    config.gates.taskspec.minimum_between_within_ratio
                ),
                minimum_severity_spearman=(
                    config.gates.taskspec.minimum_severity_spearman
                ),
                numerical_zero_tolerance=config.gates.taskspec.numerical_zero_tolerance,
            )
        )
    correctness = [
        str(source_task_by_id[str(row["selected_source_id"])])
        == binding_by_variant[str(row["variant_id"])].task
        for row in routing_rows
    ]
    routing_accuracy = float(sum(correctness) / len(correctness))
    gate_d = evaluate_gate_d(gate_d_checks)
    return (
        evaluate_gate_b(
            task_results,
            routing_accuracy=routing_accuracy,
            minimum_routing_accuracy=(
                config.gates.taskspec.minimum_max_prefix_routing_accuracy
            ),
            gate_d_passed=gate_d.passed,
        ),
        pair_rows,
        routing_rows,
    )


def build_gate_c(
    *,
    bindings: Sequence[ContextBinding],
    candidates: Sequence[CandidateRecord],
    episode_map: Mapping[tuple[str, str, str], Sequence[OracleEpisodeRecord]],
    pair_rows: Sequence[Mapping[str, Any]],
    gate_a: GateAReport,
    config: V01ExperimentConfig,
    analysis_seed_namespace: str,
) -> tuple[GateCDiagnostic, ...]:
    """Build the per-task nested Gate-C diagnostic from raw primitives."""

    binding_by_variant = {item.variant_id: item for item in bindings}
    gate_a_by_task = {item.task: item for item in gate_a.tasks}
    diagnostics: list[GateCDiagnostic] = []
    for task in config.tasks.all:
        task_bindings = tuple(item for item in bindings if item.task == task)
        nominal = next(item for item in task_bindings if item.nominal)
        shifted_bindings = tuple(item for item in task_bindings if not item.nominal)
        competent = tuple(gate_a_by_task[task].competence_set)
        if not competent:
            # empirical_competence_set always contains at least one maximizer;
            # retain the check to make a tampered Gate-A object fail closed.
            raise AnalysisContractError("Gate C received an empty competence set")
        valid_candidates = {
            item.candidate_id for item in candidates if item.task_private == task
        }
        if not set(competent) <= valid_candidates:
            raise AnalysisContractError("Gate A competence set references an unknown candidate")
        probe = np.empty((len(shifted_bindings), config.probe.banks), dtype=np.float64)
        for severity_index, binding in enumerate(shifted_bindings):
            by_bank: dict[int, float] = {}
            for row in pair_rows:
                if row["family"] != "between":
                    continue
                right = binding_by_variant[str(row["right_variant_id"])]
                if right.variant_id == binding.variant_id:
                    bank = int(row["right_bank"])
                    if bank in by_bank:
                        raise AnalysisContractError("duplicate Gate-C probe bank distance")
                    by_bank[bank] = float(row["d_phi"])
            if set(by_bank) != set(range(config.probe.banks)):
                raise AnalysisContractError("Gate-C probe bank distances are incomplete")
            probe[severity_index] = [by_bank[index] for index in range(config.probe.banks)]
        transfer = np.empty(
            (
                len(shifted_bindings),
                len(competent),
                config.oracle.episodes_per_candidate_variant,
            ),
            dtype=np.float64,
        )
        for severity_index, binding in enumerate(shifted_bindings):
            for candidate_index, candidate_id in enumerate(competent):
                transfer[severity_index, candidate_index] = paired_episode_effects(
                    _rows_as_mappings(episode_map[(task, binding.variant_id, candidate_id)]),
                    _rows_as_mappings(episode_map[(task, nominal.variant_id, candidate_id)]),
                )
        diagnostics.append(
            evaluate_gate_c(
                task=task,
                probe_distances=probe,
                paired_transfer_differences=transfer,
                resamples=config.statistics.bootstrap_resamples,
                seed=derive_bootstrap_seed(analysis_seed_namespace, task, "gate_c_nested"),
                confidence_level=config.statistics.confidence_level,
                minimum_finite_fraction=(
                    config.statistics.gate_c_minimum_finite_bootstrap_fraction
                ),
            )
        )
    return tuple(diagnostics)


def assemble_scientific_analysis(
    private_contexts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate_records: Mapping[str, Any] | Sequence[CandidateRecord | Mapping[str, Any]],
    oracle_shards: Sequence[Mapping[str, Any]],
    taskspec_matrix: Mapping[str, Any] | Any,
    *,
    config: V01ExperimentConfig,
    analysis_seed_namespace: str,
    source_task_by_id: Mapping[str, str],
    schema_view_digest_by_variant: Mapping[str, str],
    gate_d_checks: Mapping[str, bool],
) -> ScientificAnalysisResult:
    """Reconstruct all scientific outputs from raw, frozen inputs.

    ``gate_d_checks`` contains primitive booleans and is evaluated again with
    :func:`evaluate_gate_d`; a caller-supplied Gate-D ``passed`` value is never
    accepted.  Gate A/B/C likewise have no precomputed-result input.
    """

    bindings, candidates, episode_map, oracle = build_oracle_matrices(
        private_contexts,
        candidate_records,
        oracle_shards,
        config=config,
        analysis_seed_namespace=analysis_seed_namespace,
    )
    gate_a = build_gate_a(
        bindings=bindings,
        candidates=candidates,
        episode_map=episode_map,
        config=config,
        analysis_seed_namespace=analysis_seed_namespace,
    )
    gate_b, pair_rows, routing_rows = build_gate_b(
        taskspec_matrix,
        bindings=bindings,
        config=config,
        source_task_by_id=source_task_by_id,
        schema_view_digest_by_variant=schema_view_digest_by_variant,
        gate_d_checks=gate_d_checks,
    )
    gate_c = build_gate_c(
        bindings=bindings,
        candidates=candidates,
        episode_map=episode_map,
        pair_rows=pair_rows,
        gate_a=gate_a,
        config=config,
        analysis_seed_namespace=analysis_seed_namespace,
    )
    gate_d = evaluate_gate_d(gate_d_checks)
    source_count = len(source_task_by_id)
    return ScientificAnalysisResult(
        oracle=oracle,
        gate_a=gate_a,
        gate_b=gate_b,
        gate_c=gate_c,
        gate_d_dependency=gate_d.to_dict(),
        join_audit={
            "context_count": len(bindings),
            "candidate_count": len(candidates),
            "oracle_shard_count": len(episode_map),
            "oracle_episode_count": len(oracle.episode_rows),
            "oracle_aggregate_count": len(oracle.aggregates),
            "taskspec_pair_count": len(pair_rows),
            "taskspec_routing_count": len(routing_rows),
            "source_mapping_count": source_count,
            "schema_view_binding_count": len(schema_view_digest_by_variant),
            "precomputed_scientific_pass_fields_consumed": False,
        },
    )


__all__ = [
    "AnalysisContractError",
    "ContextBinding",
    "OracleMatrices",
    "REQUIRED_EXTERNAL_JOIN_EVIDENCE",
    "SCIENTIFIC_ANALYSIS_SCHEMA",
    "ScientificAnalysisResult",
    "assemble_scientific_analysis",
    "build_gate_a",
    "build_gate_b",
    "build_gate_c",
    "build_oracle_matrices",
    "parse_candidate_records",
    "parse_context_bindings",
    "required_join_evidence",
]
