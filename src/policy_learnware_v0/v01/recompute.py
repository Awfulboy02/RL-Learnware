"""Independent stratified recomputation for the v0.1 shift diagnostic.

The functions in this module deliberately do not accept aggregate matrices,
gate reports, or caller-supplied ``passed`` attestations.  Oracle summaries are
rebuilt from every immutable episode shard.  TaskSpec summaries are rebuilt
from verified primitive terms, while the raw subset frozen before results is
recomputed directly from semantic samples and the frozen Gaussian kernel.

The expensive raw encoder dependency is injected through ``semantic_rebuilder``
in :func:`load_verified_measurement_samples`.  A CLI can therefore pass assets
from :class:`~policy_learnware_v0.v01.base_runtime.VerifiedBaseRuntime` without
making this audit module depend on JAX, MuJoCo, or a particular server layout.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json, sha256_ndarrays
from ..probe.dataset import EpisodeDataset
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducedRKME
from .analysis import (
    ContextBinding,
    build_gate_a,
    build_gate_c,
    build_oracle_matrices,
)
from .config import V01ExperimentConfig
from .plans import verify_pair_plan
from .schemas import (
    EnvironmentInstanceRecord,
    MeasurementSchemaView,
    PrivateContextRecord,
    ShiftManifest,
    VariantDatasetManifest,
)
from .taskspec import (
    WeightedSemanticSample,
    direct_routing_scores,
    empirical_mmd_with_raw,
    exact_self_norm,
    taskspec_primitive_digest,
)


RECOMPUTE_AUDIT_SCHEMA = "policy-learnware.v01-stratified-recompute-audit.v0"
AUDIT_PLAN_SCHEMA = "policy-learnware.v01-stratified-audit-plan.v0"
TASKSPEC_MATRIX_SCHEMA = "policy-learnware.v01-taskspec-matrix.v0"
RTOL = 1.0e-10
ATOL = 1.0e-12


class RecomputeContractError(ValueError):
    """A raw artifact, primitive, or frozen audit selection is inconsistent."""


SemanticRebuilder = Callable[
    [str, int, EpisodeDataset, VariantDatasetManifest], WeightedSemanticSample
]


def _json_file_digest(value: Any) -> str:
    """Digest the canonical newline-terminated representation used on disk."""

    return sha256_bytes(canonical_json_bytes(value) + b"\n")


def _sample_digest(sample: WeightedSemanticSample) -> str:
    return sha256_ndarrays(sample.cache_arrays())


def _close(left: Any, right: Any, *, where: str, rtol: float, atol: float) -> None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError) as error:
        raise RecomputeContractError(f"{where} is not numeric") from error
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        raise RecomputeContractError(f"{where} is non-finite")
    if not math.isclose(left_value, right_value, rel_tol=rtol, abs_tol=atol):
        raise RecomputeContractError(
            f"{where} mismatch: stored={left_value!r}, recomputed={right_value!r}"
        )


def _strict_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise RecomputeContractError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _taskspec_payload(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        if not hasattr(value, "to_dict"):
            raise RecomputeContractError("TaskSpec matrix must be a mapping or expose to_dict()")
        value = value.to_dict()
    _strict_keys(
        value,
        {"schema", "plan_digest", "pair_rows", "routing_rows", "self_norm_rows", "clamp_count"},
        "TaskSpec matrix",
    )
    if value["schema"] != TASKSPEC_MATRIX_SCHEMA:
        raise RecomputeContractError("unsupported TaskSpec matrix schema")
    for name in ("pair_rows", "routing_rows", "self_norm_rows"):
        if not isinstance(value[name], list):
            raise RecomputeContractError(f"TaskSpec {name} must be a list")
    return value


@dataclass(frozen=True)
class MeasurementChainResult:
    """Verified raw datasets and their selector-safe semantic samples."""

    samples: Mapping[tuple[str, int], WeightedSemanticSample]
    unit_evidence: tuple[Mapping[str, Any], ...]
    contract_digest: str
    pair_plan_digest: str
    semantic_binding_method: str

    @property
    def passed(self) -> bool:
        return bool(self.unit_evidence) and all(item["passed"] is True for item in self.unit_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-measurement-chain-audit.v0",
            "passed": self.passed,
            "dataset_count": len(self.unit_evidence),
            "semantic_cache_count": len(self.unit_evidence),
            "contract_digest": self.contract_digest,
            "pair_plan_digest": self.pair_plan_digest,
            "semantic_binding_method": self.semantic_binding_method,
            "units": [dict(item) for item in self.unit_evidence],
        }


def load_verified_measurement_samples(
    measurement_root: str | Path,
    measurement_contract: Mapping[str, Any],
    pair_plan: Mapping[str, Any],
    *,
    semantic_rebuilder: SemanticRebuilder | None = None,
    trusted_semantic_cache_digests: Mapping[str, str] | None = None,
    verified_base_binding_digest: str | None = None,
    verified_base_asset_digests: Mapping[str, str] | None = None,
) -> MeasurementChainResult:
    """Verify every dataset/manifest/cache unit and return immutable samples.

    A semantic cache is accepted only when it is tied to its raw dataset by an
    independent exact rebuild *or* by a trusted immutable cache-digest map.
    Merely finding a cache beside a dataset is not a provenance proof.

    ``trusted_semantic_cache_digests`` keys use ``"<variant_id>/bank_NNN"``.
    Formal callers should prefer ``semantic_rebuilder``; the digest path is for
    a separately frozen/immutable primitive merge manifest.
    """

    root = Path(measurement_root).expanduser().resolve()
    if root.name != "measurement" or not root.is_dir():
        raise RecomputeContractError("measurement_root must be an existing measurement directory")
    _strict_keys(
        measurement_contract,
        {
            "schema", "measurement_protocol_id", "base_protocol_id", "probe_banks",
            "episodes_per_bank", "prefix_grid", "gate_prefix", "pair_plan_digest",
            "variant_ids", "schema_view_digests", "visibility",
        },
        "measurement contract",
    )
    if measurement_contract["schema"] != "policy-learnware.v01-measurement-contract.v0":
        raise RecomputeContractError("unsupported measurement contract schema")
    plan_digest = verify_pair_plan(pair_plan)
    if str(measurement_contract["pair_plan_digest"]) != plan_digest:
        raise RecomputeContractError("measurement contract does not bind the pair plan")
    contract_path = root / "measurement_contract.json"
    if not contract_path.is_file():
        raise RecomputeContractError("public measurement contract is missing")
    contract_file_digest = sha256_file(contract_path)
    if contract_path.read_bytes() != canonical_json_bytes(measurement_contract) + b"\n":
        raise RecomputeContractError("supplied contract differs from the public contract bytes")
    if semantic_rebuilder is None and trusted_semantic_cache_digests is None:
        raise RecomputeContractError(
            "semantic caches require an exact semantic_rebuilder or trusted digest bindings"
        )
    variants = tuple(str(value) for value in measurement_contract["variant_ids"])
    banks = int(measurement_contract["probe_banks"])
    episodes = int(measurement_contract["episodes_per_bank"])
    schema_digests = measurement_contract["schema_view_digests"]
    if not isinstance(schema_digests, Mapping) or set(schema_digests) != set(variants):
        raise RecomputeContractError("measurement contract schema-view coverage is invalid")
    expected_units = {
        f"{variant_id}/bank_{bank:03d}" for variant_id in variants for bank in range(banks)
    }
    if trusted_semantic_cache_digests is not None:
        if set(trusted_semantic_cache_digests) != expected_units:
            raise RecomputeContractError("trusted semantic digest map has incomplete/extra work units")

    samples: dict[tuple[str, int], WeightedSemanticSample] = {}
    evidence: list[Mapping[str, Any]] = []
    for variant_id in variants:
        for bank in range(banks):
            unit_id = f"{variant_id}/bank_{bank:03d}"
            unit_root = root / "datasets" / variant_id / f"bank_{bank:03d}"
            dataset_path = unit_root / "dataset.npz"
            manifest_path = unit_root / "manifest.json"
            cache_path = root / "semantic_cache" / variant_id / f"bank_{bank:03d}.npz"
            cache_manifest_path = (
                root / "semantic_cache" / variant_id / f"bank_{bank:03d}.json"
            )
            if not (
                dataset_path.is_file()
                and manifest_path.is_file()
                and cache_path.is_file()
                and cache_manifest_path.is_file()
            ):
                raise RecomputeContractError(f"measurement work unit is incomplete: {unit_id}")
            try:
                import json

                raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = VariantDatasetManifest.from_dict(raw_manifest)
                dataset = EpisodeDataset.load_npz(dataset_path)
            except (OSError, ValueError, TypeError) as error:
                raise RecomputeContractError(f"invalid dataset unit {unit_id}: {error}") from error
            if manifest.variant_id != variant_id or manifest.bank != bank:
                raise RecomputeContractError(f"dataset manifest path identity mismatch: {unit_id}")
            if manifest.episode_count != episodes or dataset.episode_count != episodes:
                raise RecomputeContractError(f"dataset episode count mismatch: {unit_id}")
            if manifest.transition_count != dataset.transition_count:
                raise RecomputeContractError(f"dataset transition count mismatch: {unit_id}")
            if manifest.dataset_digest != dataset.digest:
                raise RecomputeContractError(f"dataset content digest mismatch: {unit_id}")
            if tuple(int(value) for value in dataset.reset_seeds) != manifest.reset_seeds:
                raise RecomputeContractError(f"dataset reset seeds differ from manifest: {unit_id}")
            if tuple(int(value) for value in dataset.probe_seeds) != manifest.probe_seeds:
                raise RecomputeContractError(f"dataset probe seeds differ from manifest: {unit_id}")
            if manifest.base_protocol_id != str(measurement_contract["base_protocol_id"]):
                raise RecomputeContractError(f"dataset base protocol mismatch: {unit_id}")
            if manifest.measurement_contract_digest != contract_file_digest:
                raise RecomputeContractError(f"dataset contract digest mismatch: {unit_id}")
            if manifest.measurement_schema_view_digest != str(schema_digests[variant_id]):
                raise RecomputeContractError(f"dataset schema-view digest mismatch: {unit_id}")
            try:
                with np.load(cache_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"points", "weights", "episode_offsets"}:
                        raise RecomputeContractError(
                            f"semantic cache has unknown/missing members: {unit_id}"
                        )
                    if any(archive[name].dtype.hasobject for name in archive.files):
                        raise RecomputeContractError(f"semantic cache contains object arrays: {unit_id}")
                    sample = WeightedSemanticSample(
                        archive["points"], archive["weights"], archive["episode_offsets"]
                    )
            except (OSError, ValueError, TypeError) as error:
                if isinstance(error, RecomputeContractError):
                    raise
                raise RecomputeContractError(f"invalid semantic cache {unit_id}: {error}") from error
            if sample.episode_count != dataset.episode_count:
                raise RecomputeContractError(f"semantic cache episode count mismatch: {unit_id}")
            if not np.array_equal(sample.episode_offsets, dataset.episode_offsets):
                raise RecomputeContractError(f"semantic cache offsets differ from raw data: {unit_id}")
            semantic_digest = _sample_digest(sample)
            binding_methods: list[str] = []
            try:
                cache_manifest = json.loads(
                    cache_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise RecomputeContractError(
                    f"invalid semantic cache manifest {unit_id}: {error}"
                ) from error
            if not isinstance(cache_manifest, Mapping):
                raise RecomputeContractError(
                    f"semantic cache manifest is not an object: {unit_id}"
                )
            _strict_keys(
                cache_manifest,
                {
                    "schema", "variant_id", "bank", "dataset_digest",
                    "measurement_schema_view_digest", "base_binding_digest",
                    "normalization_sha256", "encoder_checkpoint_sha256",
                    "encoder_config_sha256", "cache_sha256",
                },
                f"semantic cache manifest {unit_id}",
            )
            if cache_manifest["schema"] != "policy-learnware.v01-semantic-cache-manifest.v0":
                raise RecomputeContractError(
                    f"unsupported semantic cache manifest schema: {unit_id}"
                )
            expected_cache_binding = {
                "variant_id": variant_id,
                "bank": bank,
                "dataset_digest": dataset.digest,
                "measurement_schema_view_digest": str(schema_digests[variant_id]),
                "cache_sha256": sha256_file(cache_path),
            }
            if any(
                cache_manifest.get(name) != value
                for name, value in expected_cache_binding.items()
            ):
                raise RecomputeContractError(
                    f"semantic cache manifest input/output binding mismatch: {unit_id}"
                )
            if verified_base_binding_digest is not None and (
                cache_manifest["base_binding_digest"]
                != str(verified_base_binding_digest)
            ):
                raise RecomputeContractError(
                    f"semantic cache base-runtime binding mismatch: {unit_id}"
                )
            if verified_base_asset_digests is not None:
                asset_fields = {
                    "normalization_sha256": "normalization",
                    "encoder_checkpoint_sha256": "encoder_checkpoint",
                    "encoder_config_sha256": "encoder_config",
                }
                for manifest_field, asset_name in asset_fields.items():
                    if cache_manifest[manifest_field] != verified_base_asset_digests.get(
                        asset_name
                    ):
                        raise RecomputeContractError(
                            f"semantic cache {asset_name} binding mismatch: {unit_id}"
                        )
            binding_methods.append("semantic_cache_manifest")
            if trusted_semantic_cache_digests is not None:
                if str(trusted_semantic_cache_digests[unit_id]) != semantic_digest:
                    raise RecomputeContractError(f"semantic cache trusted digest mismatch: {unit_id}")
                binding_methods.append("trusted_immutable_digest")
            if semantic_rebuilder is not None:
                rebuilt = semantic_rebuilder(variant_id, bank, dataset, manifest)
                if not isinstance(rebuilt, WeightedSemanticSample):
                    raise RecomputeContractError("semantic_rebuilder must return WeightedSemanticSample")
                if _sample_digest(rebuilt) != semantic_digest:
                    raise RecomputeContractError(f"semantic cache differs from exact raw rebuild: {unit_id}")
                for name in ("points", "weights", "episode_offsets"):
                    if not np.array_equal(getattr(rebuilt, name), getattr(sample, name)):
                        raise RecomputeContractError(
                            f"semantic cache {name} differs from exact raw rebuild: {unit_id}"
                        )
                binding_methods.append("exact_raw_rebuild")
            samples[(variant_id, bank)] = sample
            evidence.append(
                {
                    "unit_id": unit_id,
                    "passed": True,
                    "dataset_file_sha256": sha256_file(dataset_path),
                    "dataset_content_digest": dataset.digest,
                    "manifest_sha256": sha256_file(manifest_path),
                    "semantic_file_sha256": sha256_file(cache_path),
                    "semantic_manifest_sha256": sha256_file(cache_manifest_path),
                    "semantic_content_digest": semantic_digest,
                    "semantic_binding_methods": binding_methods,
                }
            )
    return MeasurementChainResult(
        samples=samples,
        unit_evidence=tuple(evidence),
        contract_digest=contract_file_digest,
        pair_plan_digest=plan_digest,
        semantic_binding_method=(
            "exact_raw_rebuild"
            if semantic_rebuilder is not None
            else "trusted_immutable_digest"
        ),
    )


@dataclass(frozen=True)
class MatrixAggregationResult:
    rebuilt_payload: Mapping[str, Any]
    primitive_digest: str
    output_digest: str
    pair_count: int
    routing_count: int
    self_term_count: int

    @property
    def passed(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-matrix-aggregation-audit.v0",
            "passed": True,
            "primitive_digest": self.primitive_digest,
            "output_digest": self.output_digest,
            "pair_count": self.pair_count,
            "routing_count": self.routing_count,
            "self_term_count": self.self_term_count,
        }


def rebuild_taskspec_aggregation(
    pair_plan: Mapping[str, Any],
    stored_matrix: Mapping[str, Any] | Any,
    *,
    trusted_primitive_digest: str | None,
    negative_tolerance: float = 1.0e-8,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> MatrixAggregationResult:
    """Rebuild all matrix summaries from bound self/cross/routing primitives.

    This function intentionally does not rerun every kernel block.  Instead it
    verifies the complete pair-plan order, reconstructs every MMD from self and
    cross terms, reconstructs every routing rank from its scores, and checks a
    trusted digest over all primitives.  The preregistered raw subset is
    independently recomputed by :func:`recompute_raw_numeric_subset`.
    """

    plan_digest = verify_pair_plan(pair_plan)
    payload = _taskspec_payload(stored_matrix)
    if str(payload["plan_digest"]) != plan_digest:
        raise RecomputeContractError("TaskSpec matrix does not bind the frozen pair plan")
    self_map: dict[tuple[str, int, int], float] = {}
    self_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["self_norm_rows"]):
        if not isinstance(raw, Mapping):
            raise RecomputeContractError("TaskSpec self-norm row must be an object")
        _strict_keys(raw, {"variant_id", "bank", "prefix", "self_norm2"}, "self-norm row")
        key = (str(raw["variant_id"]), int(raw["bank"]), int(raw["prefix"]))
        value = float(raw["self_norm2"])
        if key in self_map or not math.isfinite(value):
            raise RecomputeContractError("duplicate or non-finite TaskSpec self norm")
        self_map[key] = value
        self_rows.append({"variant_id": key[0], "bank": key[1], "prefix": key[2], "self_norm2": value})

    expected_self: set[tuple[str, int, int]] = set()
    rebuilt_pairs: list[dict[str, Any]] = []
    stored_pairs = payload["pair_rows"]
    offset = 0
    clamp_count = 0
    primitive_pairs: list[dict[str, Any]] = []
    pair_fields = {
        "family", "pair_index", "left_variant_id", "left_bank", "right_variant_id",
        "right_bank", "prefix", "raw_mmd2", "mmd2", "d_phi", "roundoff_clamped",
        "cross_term",
    }
    for family in ("within", "between"):
        records = pair_plan[family]
        for pair_index, planned in enumerate(records):
            if offset >= len(stored_pairs) or not isinstance(stored_pairs[offset], Mapping):
                raise RecomputeContractError("TaskSpec pair matrix is truncated")
            row = stored_pairs[offset]
            _strict_keys(row, pair_fields, "TaskSpec pair row")
            expected_metadata = {
                "family": family,
                "pair_index": pair_index,
                **{key: planned[key] for key in (
                    "left_variant_id", "left_bank", "right_variant_id", "right_bank", "prefix"
                )},
            }
            if any(row[key] != value for key, value in expected_metadata.items()):
                raise RecomputeContractError("TaskSpec pair rows differ from canonical pair-plan order")
            left_key = (
                str(planned["left_variant_id"]), int(planned["left_bank"]), int(planned["prefix"])
            )
            right_key = (
                str(planned["right_variant_id"]), int(planned["right_bank"]), int(planned["prefix"])
            )
            expected_self.update((left_key, right_key))
            try:
                left_norm = self_map[left_key]
                right_norm = self_map[right_key]
            except KeyError as error:
                raise RecomputeContractError("TaskSpec pair lacks a required self primitive") from error
            cross = float(row["cross_term"])
            if not math.isfinite(cross):
                raise RecomputeContractError("TaskSpec cross primitive is non-finite")
            raw_mmd2 = float(left_norm + right_norm - 2.0 * cross)
            scale = max(1.0, abs(left_norm), abs(right_norm), abs(2.0 * cross))
            if raw_mmd2 < -float(negative_tolerance) * scale:
                raise RecomputeContractError("TaskSpec primitive aggregation is materially negative")
            mmd2 = max(raw_mmd2, 0.0)
            d_phi = math.sqrt(mmd2)
            clamped = raw_mmd2 < 0.0
            for name, recomputed in (
                ("raw_mmd2", raw_mmd2), ("mmd2", mmd2), ("d_phi", d_phi)
            ):
                _close(row[name], recomputed, where=f"pair[{offset}].{name}", rtol=rtol, atol=atol)
            if row["roundoff_clamped"] is not clamped:
                raise RecomputeContractError("TaskSpec roundoff-clamp flag is inconsistent")
            clamp_count += int(clamped)
            rebuilt_pairs.append({**expected_metadata, "raw_mmd2": raw_mmd2, "mmd2": mmd2,
                                  "d_phi": d_phi, "roundoff_clamped": clamped,
                                  "cross_term": cross})
            primitive_pairs.append({**expected_metadata, "cross_term": cross})
            offset += 1
    if offset != len(stored_pairs):
        raise RecomputeContractError("TaskSpec pair matrix has extra rows")
    if set(self_map) != expected_self:
        raise RecomputeContractError("TaskSpec self primitives have incomplete/extra coverage")

    rebuilt_routing: list[dict[str, Any]] = []
    primitive_routing: list[dict[str, Any]] = []
    stored_routing = payload["routing_rows"]
    if len(stored_routing) != len(pair_plan["routing"]):
        raise RecomputeContractError("TaskSpec routing matrix has incomplete/extra rows")
    routing_fields = {"routing_index", "variant_id", "bank", "prefix", "selected_source_id", "ranking"}
    for index, (planned, row) in enumerate(zip(pair_plan["routing"], stored_routing, strict=True)):
        if not isinstance(row, Mapping):
            raise RecomputeContractError("TaskSpec routing row must be an object")
        _strict_keys(row, routing_fields, "TaskSpec routing row")
        metadata = {"routing_index": index, **dict(planned)}
        if any(row[key] != value for key, value in metadata.items()):
            raise RecomputeContractError("TaskSpec routing rows differ from canonical plan order")
        ranking = row["ranking"]
        if not isinstance(ranking, list) or not ranking:
            raise RecomputeContractError("TaskSpec routing ranking is empty")
        scores: dict[str, float] = {}
        for item in ranking:
            if not isinstance(item, Mapping):
                raise RecomputeContractError("TaskSpec routing rank is not an object")
            _strict_keys(item, {"source_id", "routing_score"}, "TaskSpec routing rank")
            source_id = str(item["source_id"])
            score = float(item["routing_score"])
            if not source_id or source_id in scores or not math.isfinite(score):
                raise RecomputeContractError("TaskSpec routing primitive is invalid")
            scores[source_id] = score
        order = sorted(scores, key=lambda source_id: (scores[source_id], source_id))
        rebuilt_ranking = [
            {"source_id": source_id, "routing_score": scores[source_id]} for source_id in order
        ]
        if ranking != rebuilt_ranking or str(row["selected_source_id"]) != order[0]:
            raise RecomputeContractError("TaskSpec routing aggregation is inconsistent")
        rebuilt_routing.append({**metadata, "selected_source_id": order[0], "ranking": rebuilt_ranking})
        primitive_routing.append({**metadata, "scores": dict(sorted(scores.items()))})
    if int(payload["clamp_count"]) != clamp_count:
        raise RecomputeContractError("TaskSpec clamp_count differs from primitive reconstruction")

    canonical_self = sorted(self_rows, key=lambda row: (row["variant_id"], row["bank"], row["prefix"]))
    primitive_payload = {
        "schema": "policy-learnware.v01-taskspec-primitives.v0",
        "plan_digest": plan_digest,
        "self_terms": canonical_self,
        "cross_terms": primitive_pairs,
        "routing_scores": primitive_routing,
    }
    primitive_digest = sha256_json(primitive_payload)
    if trusted_primitive_digest is None:
        raise RecomputeContractError(
            "full primitive aggregation requires a trusted immutable primitive digest"
        )
    if str(trusted_primitive_digest) != primitive_digest:
        raise RecomputeContractError("TaskSpec primitive digest binding mismatch")
    rebuilt_payload = {
        "schema": TASKSPEC_MATRIX_SCHEMA,
        "plan_digest": plan_digest,
        "pair_rows": rebuilt_pairs,
        "routing_rows": rebuilt_routing,
        "self_norm_rows": canonical_self,
        "clamp_count": clamp_count,
    }
    # Canonical ordering is part of the persisted matrix contract.
    if sha256_json(payload) != sha256_json(rebuilt_payload):
        raise RecomputeContractError("stored TaskSpec matrix differs from rebuilt aggregation")
    return MatrixAggregationResult(
        rebuilt_payload=rebuilt_payload,
        primitive_digest=primitive_digest,
        output_digest=sha256_json(rebuilt_payload),
        pair_count=len(rebuilt_pairs),
        routing_count=len(rebuilt_routing),
        self_term_count=len(canonical_self),
    )


@dataclass(frozen=True)
class RawNumericSubsetResult:
    task_results: tuple[Mapping[str, Any], ...]
    rtol: float = RTOL
    atol: float = ATOL

    @property
    def passed(self) -> bool:
        return bool(self.task_results) and all(item["passed"] is True for item in self.task_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-raw-numeric-subset-audit.v0",
            "passed": self.passed,
            "selection_source": "frozen_audit_plan_before_results",
            "comparison_tolerance": {"rtol": self.rtol, "atol": self.atol},
            "tasks": [dict(item) for item in self.task_results],
        }


def _validate_audit_plan(
    audit_plan: Mapping[str, Any],
    *,
    bindings: Sequence[ContextBinding],
    config: V01ExperimentConfig,
) -> tuple[Mapping[str, Mapping[str, Any]], ...]:
    _strict_keys(
        audit_plan,
        {
            "schema", "oracle_reaggregation", "taskspec_digest_coverage",
            "taskspec_aggregation_coverage", "raw_numeric_subset",
        },
        "stratified audit plan",
    )
    if audit_plan["schema"] != AUDIT_PLAN_SCHEMA:
        raise RecomputeContractError("unsupported stratified audit plan schema")
    expected_literals = {
        "oracle_reaggregation": "all_episode_rows",
        "taskspec_digest_coverage": "all_datasets_and_semantic_caches",
        "taskspec_aggregation_coverage": "all_frozen_pairs",
    }
    for name, expected in expected_literals.items():
        if audit_plan[name] != expected:
            raise RecomputeContractError(f"audit plan {name} is not frozen to {expected!r}")
    subset = audit_plan["raw_numeric_subset"]
    if not isinstance(subset, Mapping):
        raise RecomputeContractError("audit raw_numeric_subset must be an object")
    _strict_keys(subset, {"within", "between", "routing", "selection_time"}, "raw subset")
    if subset["selection_time"] != "before_results":
        raise RecomputeContractError("raw subset was not frozen before results")
    binding_map = {(item.task, item.factor): item.variant_id for item in bindings}
    expected_tasks = tuple(config.tasks.all)
    grouped: list[Mapping[str, Mapping[str, Any]]] = []
    for task in expected_tasks:
        nominal = binding_map.get((task, 1.0))
        factor_two = binding_map.get((task, 2.0))
        if nominal is None or factor_two is None:
            raise RecomputeContractError(f"{task} lacks nominal/factor-2 context binding")
        selected: dict[str, Mapping[str, Any]] = {}
        for family in ("within", "between", "routing"):
            rows = subset[family]
            if not isinstance(rows, list):
                raise RecomputeContractError(f"audit subset {family} must be a list")
            matches = [row for row in rows if isinstance(row, Mapping) and row.get("task_private") == task]
            if len(matches) != 1:
                raise RecomputeContractError(f"audit subset {family} must select one row for {task}")
            selected[family] = matches[0]
        within = selected["within"]
        between = selected["between"]
        routing = selected["routing"]
        pair_keys = {
            "task_private", "left_variant_id", "left_bank", "right_variant_id",
            "right_bank", "prefix",
        }
        _strict_keys(within, pair_keys, f"{task} audit within")
        _strict_keys(between, pair_keys, f"{task} audit between")
        _strict_keys(routing, {"task_private", "variant_id", "bank", "prefix"}, f"{task} audit routing")
        expected_within = {
            "task_private": task, "left_variant_id": nominal, "left_bank": 0,
            "right_variant_id": nominal, "right_bank": 1,
            "prefix": config.probe.gate_b_unreduced_prefix,
        }
        expected_between = {
            "task_private": task, "left_variant_id": nominal, "left_bank": 0,
            "right_variant_id": factor_two, "right_bank": 0,
            "prefix": config.probe.gate_b_unreduced_prefix,
        }
        expected_routing = {
            "task_private": task, "variant_id": factor_two, "bank": 0,
            "prefix": config.probe.max_episodes_per_bank,
        }
        if dict(within) != expected_within or dict(between) != expected_between or dict(routing) != expected_routing:
            raise RecomputeContractError(f"{task} raw subset differs from the frozen selection contract")
        grouped.append(selected)
    for family in ("within", "between", "routing"):
        if len(subset[family]) != len(expected_tasks):
            raise RecomputeContractError(f"audit subset {family} has extra task selections")
    return tuple(grouped)


def _source_digest(sources: Mapping[str, ReducedRKME]) -> str:
    rows: list[dict[str, Any]] = []
    for source_id in sorted(sources):
        source = sources[source_id]
        rows.append(
            {
                "source_id": source_id,
                "arrays_digest": sha256_ndarrays({"supports": source.supports, "beta": source.beta}),
                "bandwidth": float(source.bandwidth),
                "rkme_norm2": float(source.rkme_norm2),
                "protocol_id": str(source.protocol_id),
            }
        )
    return sha256_json(rows)


def recompute_raw_numeric_subset(
    audit_plan: Mapping[str, Any],
    *,
    bindings: Sequence[ContextBinding],
    config: V01ExperimentConfig,
    samples: Mapping[tuple[str, int], WeightedSemanticSample],
    stored_matrix: Mapping[str, Any] | Any,
    kernel: GaussianKernel,
    sources: Mapping[str, ReducedRKME],
    block_size: int = 2048,
    computation_backend: str = "numpy",
    expected_source_count: int = 6,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> RawNumericSubsetResult:
    """Independently recompute the exact frozen raw TaskSpec subset."""

    selections = _validate_audit_plan(audit_plan, bindings=bindings, config=config)
    if len(sources) != int(expected_source_count):
        raise RecomputeContractError(
            f"raw routing audit requires {expected_source_count} source RKMEs, got {len(sources)}"
        )
    payload = _taskspec_payload(stored_matrix)
    pair_rows = payload["pair_rows"]
    routing_rows = payload["routing_rows"]
    self_rows = {
        (str(row["variant_id"]), int(row["bank"]), int(row["prefix"])): float(row["self_norm2"])
        for row in payload["self_norm_rows"]
    }

    def pair_row(family: str, selection: Mapping[str, Any]) -> Mapping[str, Any]:
        keys = ("left_variant_id", "left_bank", "right_variant_id", "right_bank", "prefix")
        matches = [
            row for row in pair_rows
            if row.get("family") == family
            and all(row.get(key) == selection[key] for key in keys)
        ]
        if len(matches) != 1:
            raise RecomputeContractError(f"raw subset cannot resolve one stored {family} pair")
        return matches[0]

    task_results: list[Mapping[str, Any]] = []
    for selected in selections:
        within = selected["within"]
        between = selected["between"]
        routing = selected["routing"]
        task = str(within["task_private"])
        prefix = int(within["prefix"])
        nominal_id = str(within["left_variant_id"])
        factor_two_id = str(between["right_variant_id"])
        try:
            nominal_bank0 = samples[(nominal_id, 0)].prefix(prefix)
            nominal_bank1 = samples[(nominal_id, 1)].prefix(prefix)
            factor_two_bank0 = samples[(factor_two_id, 0)].prefix(prefix)
        except KeyError as error:
            raise RecomputeContractError(f"raw subset sample is missing for {task}") from error
        raw_samples = {
            (nominal_id, 0, prefix): nominal_bank0,
            (nominal_id, 1, prefix): nominal_bank1,
            (factor_two_id, 0, prefix): factor_two_bank0,
        }
        norms = {
            key: exact_self_norm(
                sample, kernel, block_size=block_size, computation_backend=computation_backend
            )
            for key, sample in raw_samples.items()
        }
        for key, value in norms.items():
            if key not in self_rows:
                raise RecomputeContractError(f"stored matrix lacks audited self term {key}")
            _close(self_rows[key], value, where=f"{task}.self_norm{key}", rtol=rtol, atol=atol)
        within_result = empirical_mmd_with_raw(
            nominal_bank0,
            nominal_bank1,
            kernel,
            left_norm2=norms[(nominal_id, 0, prefix)],
            right_norm2=norms[(nominal_id, 1, prefix)],
            block_size=block_size,
            computation_backend=computation_backend,
        )
        between_result = empirical_mmd_with_raw(
            nominal_bank0,
            factor_two_bank0,
            kernel,
            left_norm2=norms[(nominal_id, 0, prefix)],
            right_norm2=norms[(factor_two_id, 0, prefix)],
            block_size=block_size,
            computation_backend=computation_backend,
        )
        pair_details: dict[str, Any] = {}
        for family, selection, result in (
            ("within", within, within_result),
            ("between", between, between_result),
        ):
            stored = pair_row(family, selection)
            for name in ("left_norm2", "right_norm2", "cross_term", "raw_mmd2", "mmd2", "d_phi"):
                if name in {"left_norm2", "right_norm2"}:
                    continue
                _close(
                    stored[name], getattr(result, name), where=f"{task}.{family}.{name}",
                    rtol=rtol, atol=atol,
                )
            if stored["roundoff_clamped"] is not result.roundoff_clamped:
                raise RecomputeContractError(f"{task}.{family} clamp flag mismatch")
            pair_details[family] = {
                "cross_term": result.cross_term,
                "raw_mmd2": result.raw_mmd2,
                "mmd2": result.mmd2,
                "d_phi": result.d_phi,
                "output_digest": sha256_json(
                    {
                        "left_norm2": result.left_norm2,
                        "right_norm2": result.right_norm2,
                        "cross_term": result.cross_term,
                        "raw_mmd2": result.raw_mmd2,
                        "mmd2": result.mmd2,
                        "d_phi": result.d_phi,
                        "roundoff_clamped": result.roundoff_clamped,
                    }
                ),
            }
        routing_prefix = int(routing["prefix"])
        routing_sample = samples[(factor_two_id, 0)].prefix(routing_prefix)
        scores = direct_routing_scores(
            routing_sample,
            sources,
            kernel,
            block_size=block_size,
            computation_backend=computation_backend,
        )
        matches = [
            row for row in routing_rows
            if row.get("variant_id") == factor_two_id
            and int(row.get("bank", -1)) == 0
            and int(row.get("prefix", -1)) == routing_prefix
        ]
        if len(matches) != 1:
            raise RecomputeContractError(f"raw subset cannot resolve one routing row for {task}")
        stored_routing = matches[0]
        stored_scores = {
            str(item["source_id"]): float(item["routing_score"])
            for item in stored_routing["ranking"]
        }
        if set(stored_scores) != set(scores):
            raise RecomputeContractError(f"{task} routing source coverage mismatch")
        for source_id, score in scores.items():
            _close(
                stored_scores[source_id], score,
                where=f"{task}.routing[{source_id}]", rtol=rtol, atol=atol,
            )
        ranking = sorted(scores, key=lambda source_id: (scores[source_id], source_id))
        if str(stored_routing["selected_source_id"]) != ranking[0]:
            raise RecomputeContractError(f"{task} routing winner differs after raw recomputation")
        input_digests = {
            "nominal_bank0_prefix16": _sample_digest(nominal_bank0),
            "nominal_bank1_prefix16": _sample_digest(nominal_bank1),
            "factor2_bank0_prefix16": _sample_digest(factor_two_bank0),
            "factor2_bank0_routing_prefix": _sample_digest(routing_sample),
            "sources": _source_digest(sources),
            "kernel": sha256_json({"bandwidth": float(kernel.bandwidth)}),
        }
        task_results.append(
            {
                "task_private": task,
                "passed": True,
                "gate_prefix": prefix,
                "routing_prefix": routing_prefix,
                "self_term_count": 3,
                "cross_term_count": 2,
                "source_count": len(scores),
                "input_digests": input_digests,
                "pairs": pair_details,
                "routing": {
                    "selected_source_id": ranking[0],
                    "scores_digest": sha256_json(dict(sorted(scores.items()))),
                },
            }
        )
    return RawNumericSubsetResult(tuple(task_results), rtol=float(rtol), atol=float(atol))


@dataclass(frozen=True)
class OracleRecomputeResult:
    oracle_episodes: Mapping[str, Any]
    oracle_aggregates: Mapping[str, Any]
    gate_a: Mapping[str, Any]
    gate_c_diagnostics: tuple[Mapping[str, Any], ...]

    @property
    def passed(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-oracle-full-recompute.v0",
            "passed": True,
            "raw_source": "all_oracle_episode_shards",
            "precomputed_aggregates_or_gates_consumed": False,
            "episode_count": int(self.oracle_episodes["episode_count"]),
            "aggregate_count": int(self.oracle_aggregates["aggregate_count"]),
            "episode_matrix_digest": sha256_json(self.oracle_episodes),
            "aggregate_matrix_digest": sha256_json(self.oracle_aggregates),
            "gate_a_digest": sha256_json(self.gate_a),
            "gate_c_digest": sha256_json(list(self.gate_c_diagnostics)),
        }


def recompute_oracle_gate_a_c(
    private_contexts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    candidate_records: Mapping[str, Any] | Sequence[Mapping[str, Any] | Any],
    oracle_shards: Sequence[Mapping[str, Any]],
    *,
    rebuilt_taskspec_matrix: Mapping[str, Any],
    config: V01ExperimentConfig,
    analysis_seed_namespace: str,
) -> OracleRecomputeResult:
    """Rebuild all oracle rows/aggregates and Gate A/C from raw episode shards."""

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
    pair_rows = _taskspec_payload(rebuilt_taskspec_matrix)["pair_rows"]
    gate_c = build_gate_c(
        bindings=bindings,
        candidates=candidates,
        episode_map=episode_map,
        pair_rows=pair_rows,
        gate_a=gate_a,
        config=config,
        analysis_seed_namespace=analysis_seed_namespace,
    )
    return OracleRecomputeResult(
        oracle_episodes=oracle.episodes_dict(),
        oracle_aggregates=oracle.aggregates_dict(),
        gate_a=gate_a.to_dict(),
        gate_c_diagnostics=tuple(item.to_dict() for item in gate_c),
    )


@dataclass(frozen=True)
class StratifiedRecomputeResult:
    full_digest_coverage: Mapping[str, Any]
    full_aggregation_coverage: Mapping[str, Any]
    raw_numeric_subset: Mapping[str, Any]
    oracle_full_recompute: Mapping[str, Any]
    rtol: float = RTOL
    atol: float = ATOL

    @property
    def passed(self) -> bool:
        sections = (
            self.full_digest_coverage,
            self.full_aggregation_coverage,
            self.raw_numeric_subset,
            self.oracle_full_recompute,
        )
        return all(section.get("passed") is True for section in sections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOMPUTE_AUDIT_SCHEMA,
            "passed": self.passed,
            "audit_kind": "stratified_not_full_taskspec_raw_recompute",
            "comparison_tolerance": {"rtol": self.rtol, "atol": self.atol},
            "full_digest_coverage": dict(self.full_digest_coverage),
            "full_aggregation_coverage": dict(self.full_aggregation_coverage),
            "raw_numeric_subset": dict(self.raw_numeric_subset),
            "oracle_full_recompute": dict(self.oracle_full_recompute),
        }


def assemble_stratified_recompute_result(
    measurement_chain: MeasurementChainResult,
    aggregation: MatrixAggregationResult,
    raw_subset: RawNumericSubsetResult,
    oracle: OracleRecomputeResult,
) -> StratifiedRecomputeResult:
    """Assemble the audit only from typed executable recomputation outputs."""

    if not all(item.passed for item in (measurement_chain, aggregation, raw_subset, oracle)):
        raise RecomputeContractError("cannot assemble a passed audit from failed evidence")
    return StratifiedRecomputeResult(
        full_digest_coverage=measurement_chain.to_dict(),
        full_aggregation_coverage=aggregation.to_dict(),
        raw_numeric_subset=raw_subset.to_dict(),
        oracle_full_recompute=oracle.to_dict(),
    )


@dataclass(frozen=True)
class ExecutableEvidence:
    """Evidence produced by code execution, never an uploaded pass flag."""

    kind: str
    passed: bool
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("evidence kind must be a non-empty string")
        if type(self.passed) is not bool or not isinstance(self.details, Mapping):
            raise ValueError("invalid executable evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-executable-evidence.v0",
            "kind": self.kind,
            "passed": self.passed,
            "details": dict(self.details),
        }


def _subparser(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    for action in parser._actions:  # argparse exposes no public lookup API.
        if isinstance(action, argparse._SubParsersAction) and command in action.choices:
            return action.choices[command]
    raise RecomputeContractError(f"CLI has no registered command {command!r}")


def compute_taskspec_capability_evidence(
    parser: argparse.ArgumentParser,
    *,
    measurement_module_paths: Sequence[str | Path],
    orchestration_path: str | Path,
    orchestration_functions: Sequence[str] = (
        "_compute_taskspec",
        "_compute_taskspec_impl",
        "_resume_verified_taskspec_bundle",
    ),
) -> ExecutableEvidence:
    """Execute parser/root and source-dependency proofs for TaskSpec.

    The metric modules must not import oracle/analysis/environment-private code;
    the orchestration handler may write only ``measurement`` and may not touch
    private layout attributes.  This complements the dynamic poison test.
    """

    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        command_parser = _subparser(parser, "compute-taskspec-matrix")
        actions = {
            action.dest: {
                "required": bool(getattr(action, "required", False)),
                "options": list(action.option_strings),
            }
            for action in command_parser._actions
            if action.dest != "help"
        }
        path_capabilities = {
            name
            for name in actions
            if name.endswith("_root")
            or name in {"config", "artifacts_root", "fpo_root", "runs_root"}
        }
        if path_capabilities != {"base_artifacts_root", "measurement_root"}:
            errors.append(
                f"TaskSpec path capabilities are {sorted(path_capabilities)}"
            )
        if not all(
            actions.get(name, {}).get("required") is True
            for name in ("base_artifacts_root", "measurement_root")
        ):
            errors.append("TaskSpec roots are not both required")
        details["parser_actions"] = actions
    except Exception as error:
        errors.append(f"parser capability audit failed: {error}")

    forbidden_imports = (
        ".oracle", "v01.oracle", ".analysis", "v01.analysis",
        ".variant_env", "benchmark_private", "oracle_private",
    )
    module_evidence: list[dict[str, Any]] = []
    for raw in measurement_module_paths:
        path = Path(raw).expanduser().resolve()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append("." * node.level + (node.module or ""))
            bad = sorted(
                name
                for name in imports
                if any(fragment in name for fragment in forbidden_imports)
            )
            if bad:
                errors.append(f"{path.name} has forbidden imports {bad}")
            module_evidence.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "imports": sorted(imports),
                }
            )
        except (OSError, SyntaxError) as error:
            errors.append(f"source dependency audit failed for {path}: {error}")
    details["measurement_modules"] = module_evidence

    handler_path = Path(orchestration_path).expanduser().resolve()
    try:
        tree = ast.parse(
            handler_path.read_text(encoding="utf-8"), filename=str(handler_path)
        )
        requested_functions = tuple(str(value) for value in orchestration_functions)
        if (
            not requested_functions
            or len(set(requested_functions)) != len(requested_functions)
            or any(not value for value in requested_functions)
        ):
            raise RecomputeContractError("TaskSpec orchestration function set is invalid")
        private_attrs = {
            "benchmark_private_dir", "oracle_private_dir", "analysis_dir",
            "contexts", "candidates", "oracle_shard", "instance_record",
            "shift_manifest", "benchmark_private_root", "oracle_root",
        }
        function_evidence: dict[str, dict[str, Any]] = {}
        all_private: set[str] = set()
        all_writer_domains: set[str] = set()
        all_nested_imports: set[str] = set()
        definitions_by_name: dict[
            str, list[ast.FunctionDef | ast.AsyncFunctionDef]
        ] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions_by_name.setdefault(node.name, []).append(node)
        definitions = {
            name: nodes[0]
            for name, nodes in definitions_by_name.items()
            if len(nodes) == 1
        }
        for function_name in requested_functions:
            matches = definitions_by_name.get(function_name, [])
            if len(matches) != 1:
                errors.append(
                    "TaskSpec orchestration function missing/duplicated: "
                    f"{function_name}"
                )
                continue
            function = matches[0]
            observed_private = sorted(
                {
                    node.attr
                    for node in ast.walk(function)
                    if isinstance(node, ast.Attribute)
                    and node.attr in private_attrs
                }
            )
            writer_domains = sorted(
                {
                    node.args[0].value
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "writer"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                }
            )
            nested_imports: list[str] = []
            for node in ast.walk(function):
                if isinstance(node, ast.Import):
                    nested_imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    nested_imports.append("." * node.level + (node.module or ""))
            bad_imports = sorted(
                name
                for name in nested_imports
                if any(fragment in name for fragment in forbidden_imports)
            )
            if observed_private:
                errors.append(
                    f"TaskSpec {function_name} touches private layout/root attrs "
                    f"{observed_private}"
                )
            if set(writer_domains) - {"measurement"}:
                errors.append(
                    f"TaskSpec {function_name} writes non-measurement domains "
                    f"{writer_domains}"
                )
            if bad_imports:
                errors.append(
                    f"TaskSpec {function_name} imports private dependencies "
                    f"{bad_imports}"
                )
            all_private.update(observed_private)
            all_writer_domains.update(writer_domains)
            all_nested_imports.update(nested_imports)
            function_evidence[function_name] = {
                "writer_domains": writer_domains,
                "private_layout_or_root_attributes": observed_private,
                "nested_imports": sorted(nested_imports),
                "called_local_functions": sorted(
                    {
                        node.func.id
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in definitions
                    }
                ),
            }
        details["orchestration"] = {
            "path": str(handler_path),
            "sha256": sha256_file(handler_path),
            "functions": function_evidence,
            "audited_function_names": sorted(function_evidence),
            "writer_domains": sorted(all_writer_domains),
            "private_layout_or_root_attributes": sorted(all_private),
            "nested_imports": sorted(all_nested_imports),
        }
    except (OSError, SyntaxError, RecomputeContractError) as error:
        errors.append(f"orchestration source audit failed: {error}")
    details["errors"] = errors
    return ExecutableEvidence("taskspec_capability_and_source", not errors, details)


def compute_measurement_visibility_evidence(
    measurement_root: str | Path,
    forbidden_fields: Sequence[str],
) -> ExecutableEvidence:
    """Run recursive payload scanning and the exact artifact allowlist."""

    from .audit import assert_measurement_isolation, assert_measurement_schema_allowlist

    errors: list[str] = []
    isolation: Mapping[str, Any] = {}
    allowlist: Mapping[str, Any] = {}
    try:
        isolation = assert_measurement_isolation(measurement_root, forbidden_fields)
        allowlist = assert_measurement_schema_allowlist(measurement_root)
        if isolation.get("passed") is not True:
            errors.append("measurement forbidden-token payload scan failed")
        if allowlist.get("passed") is not True:
            errors.append("measurement exact schema allowlist failed")
    except Exception as error:
        errors.append(f"measurement visibility audit failed: {type(error).__name__}: {error}")
    return ExecutableEvidence(
        "measurement_visibility",
        not errors,
        {"payload_scan": dict(isolation), "schema_allowlist": dict(allowlist), "errors": errors},
    )


def verify_private_collection_bindings(
    *,
    frozen_root: str | Path,
    benchmark_private_root: str | Path,
    measurement_root: str | Path,
) -> ExecutableEvidence:
    """Join every public dataset to its private, live-instance attestation.

    The join key is only ``(opaque variant_id, bank, dataset_digest)``.  No
    instance or attestation digest is added to selector-visible artifacts.
    Every persisted ``passed`` member is ignored as an input: the complete
    private attestation is rebuilt from raw dataset bytes, the Gate-0 instance
    record, and immutable run bytes before canonical equality is required.
    """

    from .live_binding import verify_collection_binding_attestation

    frozen = Path(frozen_root).expanduser().resolve()
    private = Path(benchmark_private_root).expanduser().resolve()
    measurement = Path(measurement_root).expanduser().resolve()
    errors: list[str] = []
    units: list[dict[str, Any]] = []

    def load_object(path: Path, where: str) -> Mapping[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RecomputeContractError(f"{where} is not a JSON object")
        return value

    try:
        if (
            (frozen.name, private.name, measurement.name)
            != ("frozen", "benchmark_private", "measurement")
            or len({frozen.parent, private.parent, measurement.parent}) != 1
        ):
            raise RecomputeContractError(
                "collection binding roots do not belong to one experiment"
            )
        run_path = frozen / "run_manifest.json"
        run_digest = sha256_file(run_path)
        contract_path = measurement / "measurement_contract.json"
        contract = load_object(contract_path, "measurement contract")
        contexts = load_object(private / "contexts.json", "private context map")
        _strict_keys(
            contexts,
            {"schema", "experiment_id", "entries"},
            "private context map",
        )
        if contexts["schema"] != "policy-learnware.v01-private-context-map.v0":
            raise RecomputeContractError("unsupported private context map schema")
        entries = contexts["entries"]
        if not isinstance(entries, list) or not entries:
            raise RecomputeContractError("private context map is empty")

        task_by_variant: dict[str, str] = {}
        shift_by_variant: dict[str, ShiftManifest] = {}
        for raw in entries:
            if not isinstance(raw, Mapping):
                raise RecomputeContractError("private context entry is not an object")
            _strict_keys(
                raw,
                {
                    "context", "shift_manifest", "shift_manifest_digest",
                    "variant_id",
                },
                "private context entry",
            )
            context = PrivateContextRecord.from_dict(raw["context"])
            shift = ShiftManifest.from_dict(raw["shift_manifest"])
            variant_id = str(raw["variant_id"])
            if (
                variant_id in task_by_variant
                or context.private_context_id != shift.private_context_id
                or context.task != shift.task
                or raw["shift_manifest_digest"] != shift.digest
            ):
                raise RecomputeContractError(
                    f"invalid or duplicate private variant binding: {variant_id}"
                )
            task_by_variant[variant_id] = context.task
            shift_by_variant[variant_id] = shift

        variant_ids = contract.get("variant_ids")
        schema_digests = contract.get("schema_view_digests")
        probe_banks = contract.get("probe_banks")
        episodes = contract.get("episodes_per_bank")
        if (
            not isinstance(variant_ids, list)
            or not variant_ids
            or not isinstance(schema_digests, Mapping)
            or type(probe_banks) is not int
            or probe_banks <= 0
            or type(episodes) is not int
            or episodes <= 0
        ):
            raise RecomputeContractError("measurement contract coverage is invalid")
        expected_variants = {str(value) for value in variant_ids}
        if (
            len(expected_variants) != len(variant_ids)
            or expected_variants != set(task_by_variant)
            or expected_variants != set(str(key) for key in schema_digests)
        ):
            raise RecomputeContractError(
                "private contexts and public measurement variants differ"
            )

        views_by_digest: dict[str, MeasurementSchemaView] = {}
        for path in sorted((measurement / "schema_views").glob("*.json")):
            view = MeasurementSchemaView.from_dict(
                load_object(path, "measurement schema view")
            )
            if path.stem != view.schema_view_id or view.digest in views_by_digest:
                raise RecomputeContractError(
                    f"invalid or duplicate measurement schema view: {path}"
                )
            views_by_digest[view.digest] = view
        if set(views_by_digest) != {str(value) for value in schema_digests.values()}:
            raise RecomputeContractError("measurement schema-view coverage differs")

        expected_units = {
            (variant_id, bank)
            for variant_id in expected_variants
            for bank in range(probe_banks)
        }

        def indexed_units(paths: Sequence[Path], *, kind: str) -> set[tuple[str, int]]:
            result: set[tuple[str, int]] = set()
            for path in paths:
                variant_id = path.parent.name
                bank_name = path.parent.parent.name if kind == "dataset" else path.stem
                if kind == "dataset":
                    bank_name = path.parent.name
                    variant_id = path.parent.parent.name
                match = re.fullmatch(r"bank_(\d{3})", bank_name)
                if match is None:
                    raise RecomputeContractError(
                        f"invalid {kind} unit path: {path}"
                    )
                unit = (variant_id, int(match.group(1)))
                if unit in result:
                    raise RecomputeContractError(f"duplicate {kind} unit: {unit}")
                result.add(unit)
            return result

        dataset_manifests = sorted(
            (measurement / "datasets").rglob("manifest.json")
        )
        attestation_paths = sorted(
            (private / "collection_attestations").rglob("*.json")
        )
        dataset_units = indexed_units(dataset_manifests, kind="dataset")
        attestation_units = indexed_units(attestation_paths, kind="attestation")
        if dataset_units != expected_units:
            raise RecomputeContractError(
                "public dataset manifest coverage differs from frozen contract"
            )
        if attestation_units != expected_units:
            raise RecomputeContractError(
                "private collection-attestation coverage differs from datasets"
            )

        contract_digest = sha256_file(contract_path)
        for variant_id, bank in sorted(expected_units):
            manifest_path = (
                measurement / "datasets" / variant_id / f"bank_{bank:03d}"
                / "manifest.json"
            )
            dataset_path = manifest_path.with_name("dataset.npz")
            attestation_path = (
                private / "collection_attestations" / variant_id
                / f"bank_{bank:03d}.json"
            )
            task = task_by_variant[variant_id]
            instance_path = (
                private / "variants" / task / variant_id / "instance.json"
            )
            shift_path = instance_path.with_name("shift_manifest.json")
            try:
                dataset = EpisodeDataset.load_npz(dataset_path)
                sidecar = VariantDatasetManifest.from_dict(
                    load_object(manifest_path, "variant dataset manifest")
                )
                instance = EnvironmentInstanceRecord.from_dict(
                    load_object(instance_path, "Gate-0 instance record")
                )
                persisted_shift = ShiftManifest.from_dict(
                    load_object(shift_path, "persisted ShiftManifest")
                )
                expected_schema_digest = str(schema_digests[variant_id])
                if (
                    sidecar.variant_id != variant_id
                    or sidecar.bank != bank
                    or sidecar.dataset_digest != dataset.digest
                    or sidecar.episode_count != dataset.episode_count
                    or sidecar.transition_count != dataset.transition_count
                    or sidecar.reset_seeds
                    != tuple(int(value) for value in dataset.reset_seeds)
                    or sidecar.probe_seeds
                    != tuple(int(value) for value in dataset.probe_seeds)
                    or sidecar.base_protocol_id != contract.get("base_protocol_id")
                    or sidecar.measurement_contract_digest != contract_digest
                    or sidecar.measurement_schema_view_digest
                    != expected_schema_digest
                ):
                    raise RecomputeContractError(
                        "public dataset sidecar differs from raw/frozen inputs"
                    )
                if (
                    instance.variant_id != variant_id
                    or instance.shift_manifest_digest != persisted_shift.digest
                    or persisted_shift.to_dict() != shift_by_variant[variant_id].to_dict()
                    or instance.measurement_schema_view_digest
                    != expected_schema_digest
                ):
                    raise RecomputeContractError(
                        "Gate-0 instance differs from private context/schema bindings"
                    )
                view = views_by_digest[expected_schema_digest]
                attestation = load_object(
                    attestation_path, "private collection binding attestation"
                )
                verify_collection_binding_attestation(
                    attestation,
                    audited_record=instance,
                    audited_instance_record_sha256=sha256_file(instance_path),
                    dataset=dataset,
                    bank=bank,
                    expected_episode_count=episodes,
                    expected_horizon=view.horizon,
                    run_manifest_sha256=run_digest,
                )
                units.append(
                    {
                        "variant_id": variant_id,
                        "bank": bank,
                        "dataset_digest": dataset.digest,
                        "attestation_sha256": sha256_file(attestation_path),
                        "verified_instance_digest": instance.digest,
                        "passed": True,
                    }
                )
            except Exception as error:
                errors.append(
                    f"{variant_id}/bank_{bank:03d}: "
                    f"{type(error).__name__}: {error}"
                )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    return ExecutableEvidence(
        "private_collection_bindings",
        not errors and bool(units),
        {
            "join_key": ["opaque_variant_id", "bank", "dataset_digest"],
            "public_instance_digest_exposed": False,
            "verified_unit_count": len(units),
            "units": units,
            "errors": errors,
        },
    )


def compute_protocol_binding_evidence(
    *,
    frozen_root: str | Path,
    benchmark_private_root: str | Path,
    measurement_root: str | Path,
    oracle_root: str | Path,
    measurement_chain: MeasurementChainResult | None = None,
    aggregation: MatrixAggregationResult | None = None,
) -> ExecutableEvidence:
    """Recompute protocol/file/dataset/matrix bindings from persisted bytes.

    The inexpensive Gate-D path verifies the immutable semantic/primitive
    manifest and every redundant matrix representation.  Typed stratified
    recomputation results may be supplied later as additional evidence, but
    are not required: Coding Plan order evaluates Gate D before §12.1.
    """

    frozen = Path(frozen_root).expanduser().resolve()
    private = Path(benchmark_private_root).expanduser().resolve()
    measurement = Path(measurement_root).expanduser().resolve()
    oracle = Path(oracle_root).expanduser().resolve()
    roots = (frozen, private, measurement, oracle)
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)

    if tuple(path.name for path in roots) != (
        "frozen", "benchmark_private", "measurement", "oracle_private"
    ) or len({path.parent for path in roots}) != 1:
        return ExecutableEvidence(
            "protocol_digest_binding", False,
            {"checks": {}, "errors": ["domain roots do not belong to one experiment"]},
        )
    collection_binding = verify_private_collection_bindings(
        frozen_root=frozen,
        benchmark_private_root=private,
        measurement_root=measurement,
    )
    check("private_collection_attestations", collection_binding.passed)
    try:
        def obj(path: Path) -> Mapping[str, Any]:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise RecomputeContractError(f"JSON is not an object: {path}")
            return value

        run = obj(frozen / "run_manifest.json")
        file_bindings = {
            "base_ref_sha256": frozen / "base_protocol_ref.json",
            "registry_ref_sha256": frozen / "shift_registry_ref.json",
            "measurement_protocol_sha256": frozen / "measurement_protocol.json",
            "oracle_protocol_sha256": frozen / "oracle_protocol.json",
            "measurement_contract_sha256": frozen / "measurement_contract.json",
            "oracle_contract_sha256": frozen / "oracle_contract.json",
            "pair_plan_sha256": measurement / "pair_plan.json",
            "measurement_run_ref_sha256": measurement / "run_ref.json",
            "audit_plan_sha256": frozen / "audit_plan.json",
            "contexts_sha256": private / "contexts.json",
            "candidate_manifest_sha256": oracle / "candidates.json",
            "source_task_map_sha256": private / "source_task_map.json",
        }
        for field, path in file_bindings.items():
            check(
                f"run_manifest.{field}",
                path.is_file() and run.get(field) == sha256_file(path),
            )
        frozen_contract_path = frozen / "measurement_contract.json"
        public_contract_path = measurement / "measurement_contract.json"
        check(
            "measurement_contract.byte_identity",
            frozen_contract_path.read_bytes() == public_contract_path.read_bytes(),
        )
        protocol = obj(frozen / "measurement_protocol.json")
        oracle_protocol = obj(frozen / "oracle_protocol.json")
        base_ref = obj(frozen / "base_protocol_ref.json")
        contract = obj(public_contract_path)
        oracle_contract = obj(frozen / "oracle_contract.json")
        run_ref = obj(measurement / "run_ref.json")
        pair_plan = obj(measurement / "pair_plan.json")
        candidates = obj(oracle / "candidates.json")
        axes = obj(measurement / "taskspec_matrix_axes.json")
        primitive_manifest = obj(measurement / "taskspec_primitive_manifest.json")
        plan_digest = verify_pair_plan(pair_plan)
        check(
            "measurement_protocol.id",
            protocol.get("measurement_protocol_id")
            == contract.get("measurement_protocol_id")
            == run_ref.get("measurement_protocol_id"),
        )
        check(
            "oracle_protocol.id",
            oracle_protocol.get("oracle_protocol_id")
            == oracle_contract.get("oracle_protocol_id")
            == candidates.get("oracle_protocol_id"),
        )
        check(
            "measurement_protocol.file_digest",
            run_ref.get("measurement_protocol_sha256")
            == sha256_file(frozen / "measurement_protocol.json"),
        )
        check(
            "measurement_contract.file_digest",
            run_ref.get("measurement_contract_digest")
            == sha256_file(public_contract_path),
        )
        source_digests = run.get("source_digests")
        check(
            "measurement_run_ref.provenance_projection",
            run_ref.get("formal") is run.get("formal")
            and run_ref.get("git") == run.get("git")
            and run_ref.get("runtime_versions") == run.get("runtime_versions")
            and isinstance(source_digests, Mapping)
            and run_ref.get("measurement_component_digests")
            == source_digests.get("measurement")
            == protocol.get("component_digests"),
        )
        check(
            "pair_plan.digest",
            plan_digest
            == contract.get("pair_plan_digest")
            == run_ref.get("pair_plan_digest")
            == axes.get("plan_digest"),
        )
        base_projection = run_ref.get("base_protocol_ref")
        check(
            "base_protocol.projection",
            isinstance(base_projection, Mapping)
            and base_projection.get("pool_id") == base_ref.get("pool_id")
            and base_projection.get("protocol_id") == base_ref.get("protocol_id")
            and base_projection.get("protocol_draft_hash")
            == base_ref.get("protocol_draft_hash")
            and base_projection.get("binding_digest") == base_ref.get("binding_digest"),
        )
        check(
            "schema_views.map",
            run_ref.get("schema_view_digests") == contract.get("schema_view_digests")
            and set(contract.get("variant_ids", []))
            == set(contract.get("schema_view_digests", {})),
        )
        actual_schema_digests: set[str] = set()
        for path in sorted((measurement / "schema_views").glob("*.json")):
            from .schemas import MeasurementSchemaView

            view = MeasurementSchemaView.from_dict(obj(path))
            check(f"schema_view.filename:{path.name}", path.stem == view.schema_view_id)
            actual_schema_digests.add(view.digest)
        check(
            "schema_views.files",
            actual_schema_digests
            == set(str(value) for value in contract.get("schema_view_digests", {}).values()),
        )
        candidate_rows = candidates.get("candidates")
        candidate_digests = oracle_contract.get("candidate_digests")
        check(
            "oracle_candidates.coverage",
            isinstance(candidate_rows, list)
            and isinstance(candidate_digests, Mapping)
            and {str(row.get("candidate_id")) for row in candidate_rows}
            == set(candidate_digests)
            == set(oracle_contract.get("candidate_ids", [])),
        )
        if isinstance(candidate_rows, list) and isinstance(candidate_digests, Mapping):
            for row in candidate_rows:
                candidate_id = str(row.get("candidate_id"))
                check(
                    f"oracle_candidate.bundle:{candidate_id}",
                    candidate_digests.get(candidate_id) == row.get("bundle_digest"),
                )
        _strict_keys(
            primitive_manifest,
            {
                "schema", "plan_digest", "primitive_digest",
                "taskspec_matrix_sha256", "semantic_manifest_sha256",
                "semantic_content_digest",
            },
            "TaskSpec primitive manifest",
        )
        check(
            "primitive_manifest.schema",
            primitive_manifest.get("schema")
            == "policy-learnware.v01-taskspec-primitive-manifest.v0",
        )
        check(
            "primitive_manifest.plan",
            primitive_manifest.get("plan_digest") == plan_digest,
        )
        check(
            "primitive_manifest.matrix_digest",
            primitive_manifest.get("taskspec_matrix_sha256")
            == sha256_file(measurement / "taskspec_matrix_axes.json"),
        )
        check(
            "primitive_manifest.primitive_digest",
            primitive_manifest.get("primitive_digest")
            == taskspec_primitive_digest(pair_plan, axes),
        )
        semantic_manifest_map = primitive_manifest.get("semantic_manifest_sha256")
        semantic_content_map = primitive_manifest.get("semantic_content_digest")
        semantic_manifest_paths = sorted(
            (measurement / "semantic_cache").glob("*/*.json")
        )
        observed_semantic_manifest_map: dict[str, str] = {}
        observed_semantic_content_map: dict[str, str] = {}
        for path in semantic_manifest_paths:
            unit_id = f"{path.parent.name}/{path.stem}"
            observed_semantic_manifest_map[unit_id] = sha256_file(path)
            sidecar = obj(path)
            cache_path = path.with_suffix(".npz")
            dataset_path = (
                measurement
                / "datasets"
                / path.parent.name
                / path.stem
                / "dataset.npz"
            )
            dataset_manifest_path = dataset_path.with_name("manifest.json")
            valid = (
                sidecar.get("schema")
                == "policy-learnware.v01-semantic-cache-manifest.v0"
                and sidecar.get("variant_id") == path.parent.name
                and sidecar.get("bank") == int(path.stem.removeprefix("bank_"))
                and cache_path.is_file()
                and sidecar.get("cache_sha256") == sha256_file(cache_path)
                and dataset_path.is_file()
                and dataset_manifest_path.is_file()
            )
            if valid:
                dataset = EpisodeDataset.load_npz(dataset_path)
                dataset_manifest = VariantDatasetManifest.from_dict(
                    obj(dataset_manifest_path)
                )
                valid = (
                    dataset.digest == dataset_manifest.dataset_digest
                    == sidecar.get("dataset_digest")
                )
                with np.load(cache_path, allow_pickle=False) as archive:
                    sample = WeightedSemanticSample(
                        archive["points"], archive["weights"], archive["episode_offsets"]
                    )
                observed_semantic_content_map[unit_id] = _sample_digest(sample)
            check(f"semantic_chain:{unit_id}", valid)
        check(
            "primitive_manifest.semantic_manifest_digests",
            isinstance(semantic_manifest_map, Mapping)
            and dict(semantic_manifest_map) == observed_semantic_manifest_map,
        )
        check(
            "primitive_manifest.semantic_content_digests",
            isinstance(semantic_content_map, Mapping)
            and dict(semantic_content_map) == observed_semantic_content_map,
        )

        pair_rows = axes.get("pair_rows", [])
        expected_arrays = {
            "family": np.asarray(
                [0 if row["family"] == "within" else 1 for row in pair_rows],
                dtype=np.int8,
            ),
            "d_phi": np.asarray([row["d_phi"] for row in pair_rows], dtype=np.float64),
            "raw_mmd2": np.asarray(
                [row["raw_mmd2"] for row in pair_rows], dtype=np.float64
            ),
            "mmd2": np.asarray([row["mmd2"] for row in pair_rows], dtype=np.float64),
        }
        with np.load(measurement / "taskspec_matrix.npz", allow_pickle=False) as archive:
            matrix_npz_ok = set(archive.files) == set(expected_arrays) and all(
                np.array_equal(archive[name], value)
                for name, value in expected_arrays.items()
            )
        check("matrix_npz.redundant_representation", matrix_npz_ok)

        pair_fields = (
            "family", "pair_index", "left_variant_id", "left_bank",
            "right_variant_id", "right_bank", "prefix", "raw_mmd2", "mmd2",
            "d_phi", "roundoff_clamped", "cross_term",
        )
        pair_stream = io.StringIO(newline="")
        pair_writer = csv.DictWriter(
            pair_stream, fieldnames=list(pair_fields), extrasaction="raise", lineterminator="\n"
        )
        pair_writer.writeheader()
        for row in pair_rows:
            pair_writer.writerow({name: row[name] for name in pair_fields})
        check(
            "matrix_csv.redundant_representation",
            (measurement / "taskspec_matrix.csv").read_text(encoding="utf-8")
            == pair_stream.getvalue(),
        )
        routing_fields = (
            "routing_index", "variant_id", "bank", "prefix", "selected_source_id"
        )
        routing_stream = io.StringIO(newline="")
        routing_writer = csv.DictWriter(
            routing_stream,
            fieldnames=list(routing_fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        routing_writer.writeheader()
        for row in axes.get("routing_rows", []):
            routing_writer.writerow({name: row[name] for name in routing_fields})
        check(
            "routing_csv.redundant_representation",
            (measurement / "routing_matrix.csv").read_text(encoding="utf-8")
            == routing_stream.getvalue(),
        )
        if measurement_chain is not None:
            check("measurement_chain.executable", measurement_chain.passed)
            check(
                "measurement_chain.contract",
                measurement_chain.contract_digest == sha256_file(public_contract_path)
                and measurement_chain.pair_plan_digest == plan_digest,
            )
        if aggregation is not None:
            check("matrix_aggregation.executable", aggregation.passed)
            check(
                "matrix_aggregation.output_digest",
                aggregation.output_digest == sha256_json(axes),
            )
    except Exception as error:
        errors.append(f"binding audit exception: {type(error).__name__}: {error}")
    return ExecutableEvidence(
        "protocol_digest_binding",
        bool(checks) and all(checks.values()) and not errors,
        {
            "checks": checks,
            "private_collection_bindings": collection_binding.to_dict(),
            "errors": errors,
        },
    )


def compute_smoke_formal_separation_evidence(
    experiment_root: str | Path,
    *,
    peer_run_manifest_paths: Sequence[str | Path] | None = None,
) -> ExecutableEvidence:
    """Recompute formal identity and ensure smoke/formal IDs and roots differ."""

    from .config import APPROVED_FORMAL_CONFIG_DIGEST

    root = Path(experiment_root).expanduser().resolve()
    errors: list[str] = []
    peers: list[dict[str, Any]] = []
    try:
        current_path = root / "frozen" / "run_manifest.json"
        current = json.loads(current_path.read_text(encoding="utf-8"))
        expected_formal = current.get("config_digest") == APPROVED_FORMAL_CONFIG_DIGEST
        if current.get("formal") is not expected_formal:
            errors.append("current formal flag differs from approved config digest")
        if current.get("experiment_id") != root.name:
            errors.append("current experiment_id differs from artifact root")
        if expected_formal and "smoke" in root.name.lower():
            errors.append("formal run reuses a smoke-labelled experiment id")
        paths = (
            [Path(value).expanduser().resolve() for value in peer_run_manifest_paths]
            if peer_run_manifest_paths is not None
            else sorted(root.parent.glob("*/frozen/run_manifest.json"))
        )
        for path in paths:
            if path == current_path.resolve() or not path.is_file():
                continue
            peer = json.loads(path.read_text(encoding="utf-8"))
            peer_root = path.parents[1].resolve()
            peer_expected_formal = (
                peer.get("config_digest") == APPROVED_FORMAL_CONFIG_DIGEST
            )
            peers.append(
                {
                    "root": str(peer_root),
                    "experiment_id": peer.get("experiment_id"),
                    "formal": peer.get("formal"),
                }
            )
            if peer.get("formal") is not peer_expected_formal:
                errors.append(f"peer formal flag/config mismatch: {path}")
            if peer_root == root or peer.get("experiment_id") == current.get("experiment_id"):
                errors.append(f"smoke/formal run identity aliases current run: {path}")
    except Exception as error:
        errors.append(f"separation audit failed: {type(error).__name__}: {error}")
    return ExecutableEvidence(
        "smoke_formal_separation",
        not errors,
        {"experiment_root": str(root), "peer_runs": peers, "errors": errors},
    )


TaskSpecResumeRunner = Callable[[Path], Mapping[str, Any]]


def _taskspec_release_snapshot(
    measurement_root: str | Path,
    *,
    source_support_sizes: tuple[int, ...],
) -> dict[str, Any]:
    """Verify and fingerprint the five outputs plus all execution attempts."""

    from .execution_profile import (
        output_artifact_digests,
        verify_any_successful_execution_attempt,
    )

    root = Path(measurement_root).expanduser().resolve()
    if root.name != "measurement" or not root.is_dir():
        raise RecomputeContractError(
            "TaskSpec release snapshot requires an existing measurement directory"
        )
    outputs = output_artifact_digests(root)
    success = verify_any_successful_execution_attempt(
        root, source_support_sizes=source_support_sizes
    )
    attempt_id = str(success["execution_attempt_id"])
    attempt_path = root / "execution_attempts" / f"{attempt_id}.json"
    attempts = {
        path.name: sha256_file(path)
        for path in sorted((root / "execution_attempts").glob("*.json"))
    }
    if (
        success.get("status") != "SUCCESS"
        or success.get("output_artifact_sha256") != outputs
        or success.get("output_digest") != sha256_json(outputs)
        or attempt_path.name not in attempts
    ):
        raise RecomputeContractError(
            "verified SUCCESS attempt does not bind the five-output bundle"
        )
    snapshot = {
        "output_artifact_sha256": outputs,
        "output_digest": sha256_json(outputs),
        "success_attempt_id": attempt_id,
        "success_attempt_digest": str(success["attempt_digest"]),
        "success_attempt_sha256": sha256_file(attempt_path),
        "execution_attempt_artifact_sha256": attempts,
    }
    snapshot["release_digest"] = sha256_json(snapshot)
    return snapshot


def compute_oracle_poison_evidence(
    resume_runner: TaskSpecResumeRunner,
    *,
    measurement_root: str | Path,
    source_support_sizes: tuple[int, ...],
) -> ExecutableEvidence:
    """Run production zero-compute resume beside missing/poisoned oracle.

    Each isolated scenario starts from the completed public measurement tree.
    The runner receives *only* that staged measurement root.  A pass requires
    the production handler to report an immutable resume and requires every
    byte digest in the five-output bundle and all execution-attempt artifacts,
    including the verified SUCCESS self digest, to remain exactly unchanged.
    """

    errors: list[str] = []
    digests: dict[str, str] = {}
    scenarios: dict[str, Any] = {}
    baseline: Mapping[str, Any] = {}
    runner_signature = "unavailable"
    try:
        import inspect

        signature = inspect.signature(resume_runner)
        runner_signature = str(signature)
        parameters = tuple(signature.parameters.values())
        if (
            len(parameters) != 1
            or parameters[0].kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            raise RecomputeContractError(
                "TaskSpec resume runner must expose only one measurement-root capability"
            )
        measurement = Path(measurement_root).expanduser().resolve()
        if measurement.name != "measurement" or not measurement.is_dir():
            raise RecomputeContractError(
                "oracle-poison audit requires an existing measurement directory"
            )
        baseline = _taskspec_release_snapshot(
            measurement, source_support_sizes=source_support_sizes
        )

        # Run the production TaskSpec resume entrypoint in two isolated roots. A
        # sibling hard-coded/private lookup in the implementation would now see
        # the deliberately missing or poisoned oracle, while the approved
        # callable receives only the staged measurement root. Hard links keep
        # this executable audit cheap and byte-identical; copy2 is a fallback.
        def link_or_copy(source: str, destination: str) -> str:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            return destination

        with tempfile.TemporaryDirectory(
            prefix=".policy-learnware-v01-poison-", dir=measurement.parent
        ) as raw:
            temp = Path(raw)
            for label in ("missing", "poison"):
                try:
                    scenario_root = temp / label
                    staged_measurement = scenario_root / "measurement"
                    shutil.copytree(
                        measurement,
                        staged_measurement,
                        copy_function=link_or_copy,
                    )
                    scenario_oracle = scenario_root / "oracle_private"
                    if label == "poison":
                        scenario_oracle.mkdir()
                        (scenario_oracle / "tempting_gate.json").write_text(
                            json.dumps(
                                {
                                    "schema": "poison",
                                    "passed": True,
                                    "candidate_id": "attacker",
                                    "return": 1.0e99,
                                },
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    before = _taskspec_release_snapshot(
                        staged_measurement,
                        source_support_sizes=source_support_sizes,
                    )
                    if before != baseline:
                        raise RecomputeContractError(
                            f"{label} staged release differs before resume"
                        )
                    result = resume_runner(staged_measurement)
                    if not isinstance(result, Mapping) or result.get("resumed") is not True:
                        raise RecomputeContractError(
                            "TaskSpec production handler did not take strict resume path"
                        )
                    after = _taskspec_release_snapshot(
                        staged_measurement,
                        source_support_sizes=source_support_sizes,
                    )
                    if after != baseline:
                        raise RecomputeContractError(
                            f"{label} oracle scenario changed TaskSpec release bytes"
                        )
                    expected_result_binding = {
                        "execution_attempt_id": baseline["success_attempt_id"],
                        "execution_attempt_digest": baseline[
                            "success_attempt_digest"
                        ],
                        "execution_attempt_sha256": baseline[
                            "success_attempt_sha256"
                        ],
                        "taskspec_matrix_sha256": baseline[
                            "output_artifact_sha256"
                        ]["taskspec_matrix_axes.json"],
                    }
                    if any(
                        result.get(key) != value
                        for key, value in expected_result_binding.items()
                    ):
                        raise RecomputeContractError(
                            "TaskSpec resume result differs from verified release binding"
                        )
                    oracle_state_ok = (
                        not scenario_oracle.exists()
                        if label == "missing"
                        else (scenario_oracle / "tempting_gate.json").is_file()
                    )
                    if not oracle_state_ok:
                        raise RecomputeContractError(
                            f"{label} oracle scenario was mutated by TaskSpec handler"
                        )
                    digests[label] = str(after["release_digest"])
                    scenarios[label] = {
                        "strict_zero_compute_resume": True,
                        "before_release_digest": before["release_digest"],
                        "after_release_digest": after["release_digest"],
                        "oracle_state_unchanged": True,
                        "resume_result_binding": expected_result_binding,
                    }
                except Exception as error:
                    errors.append(
                        f"{label} scenario failed: {type(error).__name__}: {error}"
                    )
        if set(digests) != {"missing", "poison"}:
            errors.append("both oracle isolation scenarios did not complete")
        elif len(set(digests.values())) != 1 or next(iter(digests.values())) != baseline.get(
            "release_digest"
        ):
            errors.append("oracle scenario changed the TaskSpec release digest")
    except Exception as error:
        errors.append(f"oracle poison audit failed: {type(error).__name__}: {error}")
    return ExecutableEvidence(
        "oracle_poison_independence",
        not errors,
        {
            "runner_explicit_capabilities": ["measurement_root"],
            "runner_signature": runner_signature,
            "oracle_root_passed_to_runner": False,
            "baseline_release": dict(baseline),
            "digests": digests,
            "scenarios": scenarios,
            "errors": errors,
        },
    )


def compute_gate_d_from_evidence(
    *,
    taskspec_capability: ExecutableEvidence,
    measurement_visibility: ExecutableEvidence,
    protocol_binding: ExecutableEvidence,
    smoke_formal_separation: ExecutableEvidence,
    oracle_poison_independence: ExecutableEvidence,
) -> dict[str, Any]:
    """Compute all seven Gate-D booleans from typed executable evidence.

    This intentionally does not consume §12.1 recomputation: Gate D precedes
    ``audit-recompute`` in the approved command order.  Final report release
    requires both artifacts independently.
    """

    from .gates import evaluate_gate_d

    expected = {
        "taskspec_capability": (taskspec_capability, "taskspec_capability_and_source"),
        "measurement_visibility": (measurement_visibility, "measurement_visibility"),
        "protocol_binding": (protocol_binding, "protocol_digest_binding"),
        "smoke_formal_separation": (smoke_formal_separation, "smoke_formal_separation"),
        "oracle_poison_independence": (oracle_poison_independence, "oracle_poison_independence"),
    }
    for label, (record, kind) in expected.items():
        if not isinstance(record, ExecutableEvidence) or record.kind != kind:
            raise RecomputeContractError(f"{label} is not computed {kind!r} evidence")
    checks = {
        "measurement_artifacts_forbidden_fields_absent": measurement_visibility.passed,
        "taskspec_command_has_no_oracle_dependency": taskspec_capability.passed,
        "oracle_poison_does_not_change_taskspec_digest": oracle_poison_independence.passed,
        "context_confined_to_private_or_baseline": (
            measurement_visibility.passed and taskspec_capability.passed
        ),
        "smoke_and_formal_runs_separated": smoke_formal_separation.passed,
        "matrix_inputs_match_frozen_protocols": protocol_binding.passed,
        "visibility_artifacts_untampered": (
            measurement_visibility.passed
            and protocol_binding.passed
        ),
    }
    gate = evaluate_gate_d(checks)
    return {
        "schema": "policy-learnware.v01-executable-gate-d.v0",
        **gate.to_dict(),
        "caller_supplied_passed_attestations_consumed": False,
        "evidence": {
            label: record.to_dict() for label, (record, _) in expected.items()
        },
        "stratified_recompute_consumed": False,
    }


__all__ = [
    "ATOL",
    "AUDIT_PLAN_SCHEMA",
    "ExecutableEvidence",
    "MatrixAggregationResult",
    "MeasurementChainResult",
    "OracleRecomputeResult",
    "RECOMPUTE_AUDIT_SCHEMA",
    "RTOL",
    "RawNumericSubsetResult",
    "RecomputeContractError",
    "SemanticRebuilder",
    "StratifiedRecomputeResult",
    "TaskSpecResumeRunner",
    "assemble_stratified_recompute_result",
    "compute_gate_d_from_evidence",
    "compute_measurement_visibility_evidence",
    "compute_oracle_poison_evidence",
    "compute_protocol_binding_evidence",
    "compute_smoke_formal_separation_evidence",
    "compute_taskspec_capability_evidence",
    "load_verified_measurement_samples",
    "rebuild_taskspec_aggregation",
    "recompute_oracle_gate_a_c",
    "recompute_raw_numeric_subset",
    "taskspec_primitive_digest",
    "verify_private_collection_bindings",
]
