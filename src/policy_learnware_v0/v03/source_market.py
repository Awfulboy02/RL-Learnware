"""Evaluator-owned source championization and the v0.3 30-entry market.

Only source-anchor rollout receipts are accepted.  Training summaries and
target/oracle evidence have no field in these strict contracts and are rejected
as unknown input on deserialization.  Competence is recorded in ``OBSERVE``
mode: falling below an absolute floor never removes an anchor from the market.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from ..v02.schemas import ExecutionABIRecord, PublicMarketEntry, SourceAnchorRecord
from .pool_intake import (
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_JOB_COUNT,
    PoolIntakeCell,
    V03PoolIntakeRecord,
)

if TYPE_CHECKING:
    from .source_evaluator import SourceEvaluationAttemptRecord


class SourceMarketError(ValueError):
    """Source evidence, championization, or market publication is invalid."""


EvaluationBlock = Literal["source_selection", "source_attestation"]

SOURCE_EVALUATION_PROTOCOL_SCHEMA = "policy-learnware.v03-source-evaluation-protocol.v0"
SOURCE_EVALUATOR_RECEIPT_SCHEMA = "policy-learnware.v03-source-evaluator-receipt.v0"
SOURCE_EVALUATION_WORK_UNIT_SCHEMA = "policy-learnware.v03-source-evaluation-work-unit.v0"
SOURCE_ROLLOUT_DATASET_SCHEMA = "policy-learnware.v03-source-rollout-dataset.v0"
RAW_SOURCE_EPISODE_SHARD_SCHEMA = "policy-learnware.v03-raw-source-episode-shard.v0"
SOURCE_COMPETENCE_OBSERVATION_SCHEMA = "policy-learnware.v03-source-competence-observation.v0"
SOURCE_CHAMPION_SCHEMA = "policy-learnware.v03-source-champion.v0"
PROVISIONAL_SELECTION_SCHEMA = "policy-learnware.v03-provisional-source-selection.v0"
SOURCE_CHAMPIONIZATION_SCHEMA = "policy-learnware.v03-source-championization.v0"
DEPLOYMENT_PRIVATE_ENTRY_SCHEMA = "policy-learnware.v03-deployment-private-entry.v0"
PUBLIC_MARKET_SCHEMA = "policy-learnware.v03-public-policy-market.v0"
PRIVATE_MARKET_SCHEMA = "policy-learnware.v03-deployment-private-registry.v0"
MARKET_ID_SCHEMA = "policy-learnware.v03-policy-market-id.v0"
PRIVATE_NONCE_COMMITMENT_SCHEMA = (
    "policy-learnware.v03-private-nonce-commitment.v0"
)
MARKET_ALIAS_PROTOCOL_SCHEMA = "policy-learnware.v03-market-alias-protocol.v0"
MARKET_ALIAS_SCHEMA = "policy-learnware.v03-market-alias.v0"
MARKET_TIE_BREAK_SCHEMA = "policy-learnware.v03-market-tie-break.v0"
MARKET_ALIAS_ASSIGNMENT = "private_nonce_domain_separated_sha256"
ENGINEERING_MARKET_STATE = "ENGINEERING_CONTRACT_ONLY"
SELECTION_RULE = "max_mean_within_tolerance_then_min_std_bundle_digest_candidate_id"

PUBLIC_ENTRY_ALLOWLIST = frozenset(
    {"schema", "opaque_learnware_id", "normalized_source_competence", "tie_break_token"}
)
PUBLIC_MANIFEST_ALLOWLIST = frozenset({"schema", "policy_market_id", "entries"})

_CANDIDATE_ID = re.compile(r"^v02j-[0-9a-f]{24}$")
_OPAQUE_ID = re.compile(r"^lw-[0-9a-f]{32}$")


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise SourceMarketError(f"{where} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise SourceMarketError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SourceMarketError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result.lower() != result:
        raise SourceMarketError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SourceMarketError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _nonce(value: Any, where: str) -> str:
    return _digest(value, where)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceMarketError(f"{where} must be a positive integer")
    return value


def _seed_block(value: Any, where: str) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise SourceMarketError(f"{where} must be an iterable of reset seeds") from error
    if (
        not result
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in result)
        or result != tuple(sorted(set(result)))
    ):
        raise SourceMarketError(
            f"{where} must be sorted unique non-negative integer reset seeds"
        )
    return result


def _strict_json_file(path: Path, where: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceMarketError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SourceMarketError(f"{where} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceMarketError(f"cannot read strict JSON {where}: {error}") from error
    if not isinstance(value, dict):
        raise SourceMarketError(f"{where} must be a JSON object")
    return value


def _unit_interval(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SourceMarketError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SourceMarketError(f"{where} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class SourceEvaluationProtocol:
    """Frozen evaluator authority and source-only scientific literals."""

    intake_record_digest: str
    evaluator_implementation_digest: str
    return_contract_digest: str
    selection_seed_namespace_digest: str
    attestation_seed_namespace_digest: str
    selection_reset_seeds: tuple[int, ...]
    attestation_reset_seeds: tuple[int, ...]
    selection_episodes_per_candidate: int
    attestation_episodes_per_champion: int
    source_environment_digests: Mapping[str, str]
    competence_floors: Mapping[str, float]
    mean_tolerance: float
    lcb_z: float
    source_evaluation_protocol_digest: str | None = None
    schema: str = SOURCE_EVALUATION_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EVALUATION_PROTOCOL_SCHEMA:
            raise SourceMarketError("unsupported SourceEvaluationProtocol schema")
        for name in (
            "intake_record_digest",
            "evaluator_implementation_digest",
            "return_contract_digest",
            "selection_seed_namespace_digest",
            "attestation_seed_namespace_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.selection_seed_namespace_digest == self.attestation_seed_namespace_digest:
            raise SourceMarketError("selection and attestation require distinct seed namespaces")
        for name in ("selection_episodes_per_candidate", "attestation_episodes_per_champion"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        selection_seeds = _seed_block(self.selection_reset_seeds, "selection_reset_seeds")
        attestation_seeds = _seed_block(
            self.attestation_reset_seeds, "attestation_reset_seeds"
        )
        if len(selection_seeds) != self.selection_episodes_per_candidate:
            raise SourceMarketError(
                "selection_reset_seeds length differs from selection episode count"
            )
        if len(attestation_seeds) != self.attestation_episodes_per_champion:
            raise SourceMarketError(
                "attestation_reset_seeds length differs from attestation episode count"
            )
        if set(selection_seeds) & set(attestation_seeds):
            raise SourceMarketError("selection and attestation reset-seed blocks overlap")
        object.__setattr__(self, "selection_reset_seeds", selection_seeds)
        object.__setattr__(self, "attestation_reset_seeds", attestation_seeds)
        environments = {
            _digest(anchor, "source environment anchor"): _digest(digest, "source environment digest")
            for anchor, digest in self.source_environment_digests.items()
        }
        floors = {
            _digest(anchor, "competence-floor anchor"): _unit_interval(floor, "competence floor")
            for anchor, floor in self.competence_floors.items()
        }
        if len(environments) != EXPECTED_ANCHOR_COUNT or set(environments) != set(floors):
            raise SourceMarketError("source environments/floors must cover exactly 30 identical anchors")
        object.__setattr__(self, "source_environment_digests", MappingProxyType(dict(sorted(environments.items()))))
        object.__setattr__(self, "competence_floors", MappingProxyType(dict(sorted(floors.items()))))
        tolerance = float(self.mean_tolerance)
        z = float(self.lcb_z)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise SourceMarketError("mean_tolerance must be finite and non-negative")
        if not math.isfinite(z) or z < 0.0:
            raise SourceMarketError("lcb_z must be finite and non-negative")
        object.__setattr__(self, "mean_tolerance", tolerance)
        object.__setattr__(self, "lcb_z", z)
        expected = sha256_json(self._payload_without_digest())
        if self.source_evaluation_protocol_digest is None:
            object.__setattr__(self, "source_evaluation_protocol_digest", expected)
        elif _digest(
            self.source_evaluation_protocol_digest, "source_evaluation_protocol_digest"
        ) != expected:
            raise SourceMarketError("source evaluation protocol digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intake_record_digest": self.intake_record_digest,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "return_contract_digest": self.return_contract_digest,
            "selection_seed_namespace_digest": self.selection_seed_namespace_digest,
            "attestation_seed_namespace_digest": self.attestation_seed_namespace_digest,
            "selection_reset_seeds": list(self.selection_reset_seeds),
            "attestation_reset_seeds": list(self.attestation_reset_seeds),
            "selection_episodes_per_candidate": self.selection_episodes_per_candidate,
            "attestation_episodes_per_champion": self.attestation_episodes_per_champion,
            "source_environment_digests": dict(self.source_environment_digests),
            "competence_floors": dict(self.competence_floors),
            "mean_tolerance": self.mean_tolerance,
            "lcb_z": self.lcb_z,
            "competence_mode": "OBSERVE",
            "selection_rule": SELECTION_RULE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvaluationProtocol":
        fields = set(cls.__dataclass_fields__)
        extras = {"competence_mode", "selection_rule"}
        _strict(value, fields | extras, "SourceEvaluationProtocol")
        if value["competence_mode"] != "OBSERVE" or value["selection_rule"] != SELECTION_RULE:
            raise SourceMarketError("source evaluation protocol mode/rule drifted")
        return cls(
            **{
                name: (
                    tuple(value[name])
                    if name in {"selection_reset_seeds", "attestation_reset_seeds"}
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class SourceEvaluationWorkUnit:
    """Private evaluator instruction for one source-only rollout block."""

    source_evaluation_protocol_digest: str
    intake_record_digest: str
    intake_cell_digest: str
    block: EvaluationBlock
    seed_namespace_digest: str
    candidate_id: str
    source_anchor_id: str
    attempt_number: int
    attempt_digest: str
    bundle_digest: str
    bundle_path: str
    outer_iteration: int
    environment_steps: int
    anchor_manifest_path: str
    anchor_manifest_digest: str
    anchor_runtime_digest: str
    source_environment_digest: str
    evaluator_implementation_digest: str
    return_contract_digest: str
    execution_abi: ExecutionABIRecord
    reset_seeds: tuple[int, ...]
    work_unit_digest: str | None = None
    schema: str = SOURCE_EVALUATION_WORK_UNIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EVALUATION_WORK_UNIT_SCHEMA:
            raise SourceMarketError("unsupported SourceEvaluationWorkUnit schema")
        for name in (
            "source_evaluation_protocol_digest",
            "intake_record_digest",
            "intake_cell_digest",
            "seed_namespace_digest",
            "source_anchor_id",
            "attempt_digest",
            "bundle_digest",
            "anchor_manifest_digest",
            "anchor_runtime_digest",
            "source_environment_digest",
            "evaluator_implementation_digest",
            "return_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.block not in {"source_selection", "source_attestation"}:
            raise SourceMarketError("work-unit block must be source_selection or source_attestation")
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("work-unit candidate_id is not an intake job ID")
        object.__setattr__(self, "attempt_number", _positive_int(self.attempt_number, "attempt_number"))
        object.__setattr__(self, "outer_iteration", _positive_int(self.outer_iteration, "outer_iteration"))
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        object.__setattr__(self, "bundle_path", _nonempty(self.bundle_path, "bundle_path"))
        object.__setattr__(
            self,
            "anchor_manifest_path",
            _nonempty(self.anchor_manifest_path, "anchor_manifest_path"),
        )
        if not isinstance(self.execution_abi, ExecutionABIRecord):
            raise SourceMarketError("source evaluation work unit requires a typed execution ABI")
        object.__setattr__(self, "reset_seeds", _seed_block(self.reset_seeds, "reset_seeds"))
        expected = sha256_json(self._payload_without_digest())
        if self.work_unit_digest is None:
            object.__setattr__(self, "work_unit_digest", expected)
        elif _digest(self.work_unit_digest, "work_unit_digest") != expected:
            raise SourceMarketError("source evaluation work-unit digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "intake_record_digest": self.intake_record_digest,
            "intake_cell_digest": self.intake_cell_digest,
            "block": self.block,
            "seed_namespace_digest": self.seed_namespace_digest,
            "candidate_id": self.candidate_id,
            "source_anchor_id": self.source_anchor_id,
            "attempt_number": self.attempt_number,
            "attempt_digest": self.attempt_digest,
            "bundle_digest": self.bundle_digest,
            "bundle_path": self.bundle_path,
            "outer_iteration": self.outer_iteration,
            "environment_steps": self.environment_steps,
            "anchor_manifest_path": self.anchor_manifest_path,
            "anchor_manifest_digest": self.anchor_manifest_digest,
            "anchor_runtime_digest": self.anchor_runtime_digest,
            "source_environment_digest": self.source_environment_digest,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "return_contract_digest": self.return_contract_digest,
            "execution_abi": self.execution_abi.to_dict(),
            "reset_seeds": list(self.reset_seeds),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "work_unit_digest": self.work_unit_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvaluationWorkUnit":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceEvaluationWorkUnit")
        try:
            abi = ExecutionABIRecord.from_dict(value["execution_abi"])
        except (TypeError, ValueError) as error:
            raise SourceMarketError("work unit has an invalid execution ABI") from error
        return cls(
            **{
                name: (
                    tuple(value[name])
                    if name == "reset_seeds"
                    else abi
                    if name == "execution_abi"
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class RawSourceEpisodeShard:
    """Evaluator-emitted per-episode source returns before receipt aggregation."""

    work_unit_digest: str
    attempt_record_digest: str
    validated_binding_digest: str
    evaluation_attempt_number: int
    block: EvaluationBlock
    candidate_id: str
    runtime_digest: str
    reset_seeds: tuple[int, ...]
    raw_episode_returns: tuple[float, ...]
    normalized_returns: tuple[float, ...]
    return_contract_digest: str
    episode_shard_digest: str | None = None
    schema: str = RAW_SOURCE_EPISODE_SHARD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RAW_SOURCE_EPISODE_SHARD_SCHEMA:
            raise SourceMarketError("unsupported RawSourceEpisodeShard schema")
        for name in (
            "work_unit_digest",
            "attempt_record_digest",
            "validated_binding_digest",
            "runtime_digest",
            "return_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "evaluation_attempt_number",
            _positive_int(self.evaluation_attempt_number, "evaluation_attempt_number"),
        )
        if self.block not in {"source_selection", "source_attestation"}:
            raise SourceMarketError("raw source shard has an invalid block")
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("raw source shard candidate_id is invalid")
        seeds = _seed_block(self.reset_seeds, "raw source shard reset_seeds")
        raw = tuple(self.raw_episode_returns)
        normalized = tuple(self.normalized_returns)
        if len(raw) != len(seeds) or len(normalized) != len(seeds):
            raise SourceMarketError("raw source shard vectors differ from its reset-seed block")
        parsed_raw: list[float] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise SourceMarketError("raw source episode returns must be numeric")
            item = float(value)
            if not math.isfinite(item):
                raise SourceMarketError("raw source episode returns must be finite")
            parsed_raw.append(item)
        parsed_normalized = tuple(
            _unit_interval(value, "normalized source return") for value in normalized
        )
        object.__setattr__(self, "reset_seeds", seeds)
        object.__setattr__(self, "raw_episode_returns", tuple(parsed_raw))
        object.__setattr__(self, "normalized_returns", parsed_normalized)
        expected = sha256_json(self._payload_without_digest())
        if self.episode_shard_digest is None:
            object.__setattr__(self, "episode_shard_digest", expected)
        elif _digest(self.episode_shard_digest, "episode_shard_digest") != expected:
            raise SourceMarketError("raw source episode-shard digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "work_unit_digest": self.work_unit_digest,
            "attempt_record_digest": self.attempt_record_digest,
            "validated_binding_digest": self.validated_binding_digest,
            "evaluation_attempt_number": self.evaluation_attempt_number,
            "block": self.block,
            "candidate_id": self.candidate_id,
            "runtime_digest": self.runtime_digest,
            "reset_seeds": list(self.reset_seeds),
            "raw_episode_returns": list(self.raw_episode_returns),
            "normalized_returns": list(self.normalized_returns),
            "return_contract_digest": self.return_contract_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "episode_shard_digest": self.episode_shard_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RawSourceEpisodeShard":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "RawSourceEpisodeShard")
        return cls(
            **{
                name: (
                    tuple(value[name])
                    if name in {"reset_seeds", "raw_episode_returns", "normalized_returns"}
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class EvaluatorSourceReceipt:
    """One evaluator-produced, source-only episode block for one candidate."""

    source_evaluation_protocol_digest: str
    intake_record_digest: str
    intake_cell_digest: str
    block: EvaluationBlock
    seed_namespace_digest: str
    candidate_id: str
    source_anchor_id: str
    bundle_digest: str
    source_environment_digest: str
    evaluator_implementation_digest: str
    return_contract_digest: str
    work_unit_digest: str
    attempt_record_digest: str
    validated_binding_digest: str
    evaluation_attempt_number: int
    runtime_digest: str
    raw_episode_shard_digest: str
    dataset_digest: str
    reset_seeds: tuple[int, ...]
    normalized_returns: tuple[float, ...]
    receipt_digest: str | None = None
    schema: str = SOURCE_EVALUATOR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EVALUATOR_RECEIPT_SCHEMA:
            raise SourceMarketError("unsupported EvaluatorSourceReceipt schema")
        for name in (
            "source_evaluation_protocol_digest",
            "intake_record_digest",
            "intake_cell_digest",
            "seed_namespace_digest",
            "source_anchor_id",
            "bundle_digest",
            "source_environment_digest",
            "evaluator_implementation_digest",
            "return_contract_digest",
            "work_unit_digest",
            "attempt_record_digest",
            "validated_binding_digest",
            "runtime_digest",
            "raw_episode_shard_digest",
            "dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "evaluation_attempt_number",
            _positive_int(self.evaluation_attempt_number, "evaluation_attempt_number"),
        )
        if self.block not in {"source_selection", "source_attestation"}:
            raise SourceMarketError("receipt block must be source_selection or source_attestation")
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("receipt candidate_id is not an intake job ID")
        seeds = _seed_block(self.reset_seeds, "receipt reset_seeds")
        returns = tuple(self.normalized_returns)
        if len(seeds) != len(returns):
            raise SourceMarketError("receipt must contain aligned source episode seeds/returns")
        parsed_returns = tuple(_unit_interval(value, "normalized source return") for value in returns)
        object.__setattr__(self, "reset_seeds", seeds)
        object.__setattr__(self, "normalized_returns", parsed_returns)
        expected = sha256_json(self._payload_without_digest())
        if self.receipt_digest is None:
            object.__setattr__(self, "receipt_digest", expected)
        elif _digest(self.receipt_digest, "receipt_digest") != expected:
            raise SourceMarketError("source receipt digest does not match contents")

    @property
    def episode_count(self) -> int:
        return len(self.normalized_returns)

    @property
    def mean(self) -> float:
        return float(np.mean(np.asarray(self.normalized_returns, dtype=np.float64)))

    @property
    def std(self) -> float:
        return float(np.std(np.asarray(self.normalized_returns, dtype=np.float64), ddof=0))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "intake_record_digest": self.intake_record_digest,
            "intake_cell_digest": self.intake_cell_digest,
            "block": self.block,
            "seed_namespace_digest": self.seed_namespace_digest,
            "candidate_id": self.candidate_id,
            "source_anchor_id": self.source_anchor_id,
            "bundle_digest": self.bundle_digest,
            "source_environment_digest": self.source_environment_digest,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "return_contract_digest": self.return_contract_digest,
            "work_unit_digest": self.work_unit_digest,
            "attempt_record_digest": self.attempt_record_digest,
            "validated_binding_digest": self.validated_binding_digest,
            "evaluation_attempt_number": self.evaluation_attempt_number,
            "runtime_digest": self.runtime_digest,
            "raw_episode_shard_digest": self.raw_episode_shard_digest,
            "dataset_digest": self.dataset_digest,
            "reset_seeds": list(self.reset_seeds),
            "normalized_returns": list(self.normalized_returns),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluatorSourceReceipt":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "EvaluatorSourceReceipt")
        return cls(
            **{
                name: tuple(value[name]) if name in {"reset_seeds", "normalized_returns"} else value[name]
                for name in fields
            }
        )


def build_source_evaluation_work_unit(
    intake: V03PoolIntakeRecord,
    protocol: SourceEvaluationProtocol,
    candidate_id: str,
    *,
    block: EvaluationBlock,
    anchor_manifest_path: str | Path,
    execution_abi: ExecutionABIRecord,
) -> SourceEvaluationWorkUnit:
    """Bind one intake cell, canonical source manifest, and frozen seed block."""

    if not isinstance(intake, V03PoolIntakeRecord) or intake.pool_state != "POOL_READY":
        raise SourceMarketError("work-unit construction requires typed POOL_READY intake")
    if not isinstance(protocol, SourceEvaluationProtocol):
        raise SourceMarketError("work-unit construction requires a typed source protocol")
    if protocol.intake_record_digest != intake.intake_record_digest:
        raise SourceMarketError("work-unit protocol belongs to another intake")
    if candidate_id not in intake.cells:
        raise SourceMarketError("work-unit candidate is absent from the exact-90 intake")
    if block not in {"source_selection", "source_attestation"}:
        raise SourceMarketError("work-unit block must be source_selection or source_attestation")
    supplied_manifest_path = Path(anchor_manifest_path).expanduser()
    if not supplied_manifest_path.is_absolute():
        raise SourceMarketError("source anchor manifest path must be absolute")
    if supplied_manifest_path.is_symlink():
        raise SourceMarketError("source anchor manifest path must be a non-symlink file")
    try:
        manifest_path = supplied_manifest_path.resolve(strict=True)
    except OSError as error:
        raise SourceMarketError("source anchor manifest path does not exist") from error
    if not manifest_path.is_file():
        raise SourceMarketError("source anchor manifest path must be a non-symlink file")
    anchor_manifest = _strict_json_file(manifest_path, "source anchor manifest")
    forbidden = {"training_summary", "target_evidence", "oracle_return", "target", "oracle"}
    overlap = forbidden & set(anchor_manifest)
    if overlap:
        raise SourceMarketError(
            f"source anchor manifest contains forbidden non-source fields: {sorted(overlap)}"
        )
    required = {
        "schema",
        "anchor_id",
        "environment_instance_digest",
        "axis_binding_digest",
        "runtime",
        "runtime_digest",
        "manifest_digest",
    }
    if not required.issubset(anchor_manifest):
        raise SourceMarketError(
            f"source anchor manifest lacks evaluator bindings: {sorted(required-set(anchor_manifest))}"
        )
    if anchor_manifest["schema"] != "policy-learnware.v02-anchor-manifest.v0":
        raise SourceMarketError("unsupported source anchor-manifest schema")
    manifest_digest = _digest(anchor_manifest["manifest_digest"], "anchor manifest digest")
    if manifest_digest != sha256_json(
        {name: item for name, item in anchor_manifest.items() if name != "manifest_digest"}
    ):
        raise SourceMarketError("source anchor manifest self digest does not match contents")
    runtime = anchor_manifest["runtime"]
    if not isinstance(runtime, Mapping):
        raise SourceMarketError("source anchor runtime must be a mapping")
    runtime_digest = _digest(anchor_manifest["runtime_digest"], "anchor runtime digest")
    if runtime_digest != sha256_json(runtime):
        raise SourceMarketError("source anchor runtime digest does not match contents")
    try:
        anchor = SourceAnchorRecord(
            anchor_id=anchor_manifest["anchor_id"],
            environment_instance_digest=anchor_manifest["environment_instance_digest"],
            axis_binding_digest=anchor_manifest["axis_binding_digest"],
        )
    except (TypeError, ValueError) as error:
        raise SourceMarketError("source anchor identity is not canonical") from error
    cell = intake.cells[candidate_id]
    expected_environment = protocol.source_environment_digests.get(cell.source_anchor_id)
    if (
        anchor.anchor_id != cell.source_anchor_id
        or expected_environment is None
        or anchor.environment_instance_digest != expected_environment
    ):
        raise SourceMarketError("source anchor manifest differs from intake/protocol binding")
    if not isinstance(execution_abi, ExecutionABIRecord):
        raise SourceMarketError("work-unit construction requires a typed execution ABI")
    namespace = (
        protocol.selection_seed_namespace_digest
        if block == "source_selection"
        else protocol.attestation_seed_namespace_digest
    )
    seeds = (
        protocol.selection_reset_seeds
        if block == "source_selection"
        else protocol.attestation_reset_seeds
    )
    return SourceEvaluationWorkUnit(
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        intake_record_digest=intake.intake_record_digest,
        intake_cell_digest=cell.intake_cell_digest,
        block=block,
        seed_namespace_digest=namespace,
        candidate_id=cell.job_id,
        source_anchor_id=cell.source_anchor_id,
        attempt_number=cell.attempt_number,
        attempt_digest=cell.attempt_digest,
        bundle_digest=cell.bundle_digest,
        bundle_path=cell.bundle_path,
        outer_iteration=cell.outer_iteration,
        environment_steps=cell.environment_steps,
        anchor_manifest_path=str(manifest_path),
        anchor_manifest_digest=manifest_digest,
        anchor_runtime_digest=runtime_digest,
        source_environment_digest=expected_environment,
        evaluator_implementation_digest=protocol.evaluator_implementation_digest,
        return_contract_digest=protocol.return_contract_digest,
        execution_abi=execution_abi,
        reset_seeds=seeds,
    )


def receipt_from_source_episode_shard(
    work_unit: SourceEvaluationWorkUnit,
    episode_shard: RawSourceEpisodeShard,
    attempt: "SourceEvaluationAttemptRecord",
) -> EvaluatorSourceReceipt:
    """Create a receipt from an evaluator-owned raw episode shard.

    The shard carries raw per-episode outcomes and the corresponding projection
    of the frozen return contract.  A caller-provided aggregate or normalized
    vector is never accepted by this function.
    """

    # Local import avoids a module cycle while retaining a real runtime type
    # boundary: source_evaluator owns the attempt record and imports this module.
    from .source_evaluator import SourceEvaluationAttemptRecord

    if not isinstance(work_unit, SourceEvaluationWorkUnit):
        raise SourceMarketError("receipt construction requires a typed work unit")
    if not isinstance(episode_shard, RawSourceEpisodeShard):
        raise SourceMarketError("receipt construction requires a typed raw episode shard")
    if not isinstance(attempt, SourceEvaluationAttemptRecord) or attempt.state != "SUCCEEDED":
        raise SourceMarketError("receipt construction requires a successful typed attempt")
    attempt_raw = tuple(row.raw_return for row in attempt.episodes)
    attempt_normalized = tuple(row.normalized_return for row in attempt.episodes)
    if (
        attempt.work_unit_digest != work_unit.work_unit_digest
        or attempt.candidate_id != work_unit.candidate_id
        or attempt.block != work_unit.block
        or attempt.evaluator_implementation_digest
        != work_unit.evaluator_implementation_digest
        or attempt.runtime_digest != work_unit.anchor_runtime_digest
        or attempt.return_contract_digest != work_unit.return_contract_digest
        or attempt.expected_reset_seeds != work_unit.reset_seeds
        or episode_shard.work_unit_digest != work_unit.work_unit_digest
        or episode_shard.attempt_record_digest != attempt.attempt_record_digest
        or episode_shard.validated_binding_digest != attempt.validated_binding_digest
        or episode_shard.evaluation_attempt_number != attempt.evaluation_attempt_number
        or episode_shard.block != work_unit.block
        or episode_shard.candidate_id != work_unit.candidate_id
        or episode_shard.runtime_digest != attempt.runtime_digest
        or episode_shard.reset_seeds != work_unit.reset_seeds
        or episode_shard.raw_episode_returns != attempt_raw
        or episode_shard.normalized_returns != attempt_normalized
        or episode_shard.return_contract_digest != work_unit.return_contract_digest
    ):
        raise SourceMarketError(
            "successful attempt/raw episode shard differs from its source work unit"
        )
    dataset_digest = sha256_json(
        {
            "schema": SOURCE_ROLLOUT_DATASET_SCHEMA,
            "work_unit_digest": work_unit.work_unit_digest,
            "attempt_record_digest": attempt.attempt_record_digest,
            "validated_binding_digest": attempt.validated_binding_digest,
            "evaluation_attempt_number": attempt.evaluation_attempt_number,
            "raw_episode_shard_digest": episode_shard.episode_shard_digest,
        }
    )
    return EvaluatorSourceReceipt(
        source_evaluation_protocol_digest=work_unit.source_evaluation_protocol_digest,
        intake_record_digest=work_unit.intake_record_digest,
        intake_cell_digest=work_unit.intake_cell_digest,
        block=work_unit.block,
        seed_namespace_digest=work_unit.seed_namespace_digest,
        candidate_id=work_unit.candidate_id,
        source_anchor_id=work_unit.source_anchor_id,
        bundle_digest=work_unit.bundle_digest,
        source_environment_digest=work_unit.source_environment_digest,
        evaluator_implementation_digest=work_unit.evaluator_implementation_digest,
        return_contract_digest=work_unit.return_contract_digest,
        work_unit_digest=work_unit.work_unit_digest,
        attempt_record_digest=attempt.attempt_record_digest,
        validated_binding_digest=attempt.validated_binding_digest,
        evaluation_attempt_number=attempt.evaluation_attempt_number,
        runtime_digest=attempt.runtime_digest,
        raw_episode_shard_digest=episode_shard.episode_shard_digest,
        dataset_digest=dataset_digest,
        reset_seeds=work_unit.reset_seeds,
        normalized_returns=episode_shard.normalized_returns,
    )


@dataclass(frozen=True)
class SourceCompetenceObservation:
    source_anchor_id: str
    candidate_id: str
    attestation_receipt_digest: str
    episode_count: int
    mean: float
    std: float
    lcb: float
    normalized_competence: float
    competence_floor: float
    passed: bool
    mode: str = "OBSERVE"
    observation_digest: str | None = None
    schema: str = SOURCE_COMPETENCE_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_COMPETENCE_OBSERVATION_SCHEMA or self.mode != "OBSERVE":
            raise SourceMarketError("source competence must use OBSERVE schema")
        object.__setattr__(self, "source_anchor_id", _digest(self.source_anchor_id, "source_anchor_id"))
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("competence candidate_id is invalid")
        object.__setattr__(
            self,
            "attestation_receipt_digest",
            _digest(self.attestation_receipt_digest, "attestation_receipt_digest"),
        )
        object.__setattr__(self, "episode_count", _positive_int(self.episode_count, "episode_count"))
        mean = _unit_interval(self.mean, "competence mean")
        std = float(self.std)
        lcb = float(self.lcb)
        if not math.isfinite(std) or std < 0.0 or not math.isfinite(lcb):
            raise SourceMarketError("competence std/lcb must be finite and std non-negative")
        normalized = _unit_interval(self.normalized_competence, "normalized_competence")
        floor = _unit_interval(self.competence_floor, "competence_floor")
        if type(self.passed) is not bool or self.passed != (normalized >= floor):
            raise SourceMarketError("competence passed flag disagrees with observed floor")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "lcb", lcb)
        object.__setattr__(self, "normalized_competence", normalized)
        object.__setattr__(self, "competence_floor", floor)
        expected = sha256_json(self._payload_without_digest())
        if self.observation_digest is None:
            object.__setattr__(self, "observation_digest", expected)
        elif _digest(self.observation_digest, "observation_digest") != expected:
            raise SourceMarketError("competence observation digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "source_anchor_id": self.source_anchor_id,
            "candidate_id": self.candidate_id,
            "attestation_receipt_digest": self.attestation_receipt_digest,
            "episode_count": self.episode_count,
            "mean": self.mean,
            "std": self.std,
            "lcb": self.lcb,
            "normalized_competence": self.normalized_competence,
            "competence_floor": self.competence_floor,
            "passed": self.passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "observation_digest": self.observation_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceCompetenceObservation":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceCompetenceObservation")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class SourceChampion:
    source_anchor_id: str
    candidate_id: str
    seed: int
    intake_cell_digest: str
    bundle_digest: str
    bundle_path: str
    outer_iteration: int
    environment_steps: int
    selection_receipt_digest: str
    attestation_receipt_digest: str
    competence: SourceCompetenceObservation
    champion_digest: str | None = None
    schema: str = SOURCE_CHAMPION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_CHAMPION_SCHEMA:
            raise SourceMarketError("unsupported SourceChampion schema")
        object.__setattr__(self, "source_anchor_id", _digest(self.source_anchor_id, "source_anchor_id"))
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("champion candidate_id is invalid")
        for name in (
            "intake_cell_digest",
            "bundle_digest",
            "selection_receipt_digest",
            "attestation_receipt_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.seed not in {0, 1, 2}:
            raise SourceMarketError("champion seed must be 0/1/2")
        _nonempty(self.bundle_path, "champion bundle_path")
        object.__setattr__(self, "outer_iteration", _positive_int(self.outer_iteration, "outer_iteration"))
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        if not isinstance(self.competence, SourceCompetenceObservation):
            raise SourceMarketError("champion requires a typed competence observation")
        if (
            self.competence.source_anchor_id != self.source_anchor_id
            or self.competence.candidate_id != self.candidate_id
            or self.competence.attestation_receipt_digest != self.attestation_receipt_digest
        ):
            raise SourceMarketError("champion competence belongs to another candidate")
        expected = sha256_json(self._payload_without_digest())
        if self.champion_digest is None:
            object.__setattr__(self, "champion_digest", expected)
        elif _digest(self.champion_digest, "champion_digest") != expected:
            raise SourceMarketError("champion digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_anchor_id": self.source_anchor_id,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "intake_cell_digest": self.intake_cell_digest,
            "bundle_digest": self.bundle_digest,
            "bundle_path": self.bundle_path,
            "outer_iteration": self.outer_iteration,
            "environment_steps": self.environment_steps,
            "selection_receipt_digest": self.selection_receipt_digest,
            "attestation_receipt_digest": self.attestation_receipt_digest,
            "competence": self.competence.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "champion_digest": self.champion_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceChampion":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceChampion")
        return cls(
            **{
                name: (
                    SourceCompetenceObservation.from_dict(value[name])
                    if name == "competence"
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class ProvisionalSourceSelection:
    """Outcome-blind handoff from 90-way source selection to attestation."""

    intake_record_digest: str
    source_evaluation_protocol_digest: str
    selection_receipt_index_digest: str
    selected_candidate_ids: Mapping[str, str]
    selected_receipt_digests: Mapping[str, str]
    selected_work_unit_digests: Mapping[str, str]
    provisional_selection_digest: str | None = None
    schema: str = PROVISIONAL_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROVISIONAL_SELECTION_SCHEMA:
            raise SourceMarketError("unsupported ProvisionalSourceSelection schema")
        for name in (
            "intake_record_digest",
            "source_evaluation_protocol_digest",
            "selection_receipt_index_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        selected = dict(self.selected_candidate_ids)
        receipts = dict(self.selected_receipt_digests)
        units = dict(self.selected_work_unit_digests)
        if (
            len(selected) != EXPECTED_ANCHOR_COUNT
            or set(selected) != set(receipts)
            or set(selected) != set(units)
            or len(set(selected.values())) != EXPECTED_ANCHOR_COUNT
        ):
            raise SourceMarketError("provisional selection must bind one unique candidate per 30 anchors")
        for anchor, candidate in selected.items():
            _digest(anchor, "provisional source anchor")
            if not _CANDIDATE_ID.fullmatch(_nonempty(candidate, "provisional candidate_id")):
                raise SourceMarketError("provisional candidate_id is invalid")
            receipts[anchor] = _digest(receipts[anchor], "selected receipt digest")
            units[anchor] = _digest(units[anchor], "selected work-unit digest")
        object.__setattr__(self, "selected_candidate_ids", MappingProxyType(dict(sorted(selected.items()))))
        object.__setattr__(self, "selected_receipt_digests", MappingProxyType(dict(sorted(receipts.items()))))
        object.__setattr__(self, "selected_work_unit_digests", MappingProxyType(dict(sorted(units.items()))))
        expected = sha256_json(self._payload_without_digest())
        if self.provisional_selection_digest is None:
            object.__setattr__(self, "provisional_selection_digest", expected)
        elif _digest(self.provisional_selection_digest, "provisional_selection_digest") != expected:
            raise SourceMarketError("provisional selection digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intake_record_digest": self.intake_record_digest,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "selection_receipt_index_digest": self.selection_receipt_index_digest,
            "selected_candidate_ids": dict(self.selected_candidate_ids),
            "selected_receipt_digests": dict(self.selected_receipt_digests),
            "selected_work_unit_digests": dict(self.selected_work_unit_digests),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "provisional_selection_digest": self.provisional_selection_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvisionalSourceSelection":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "ProvisionalSourceSelection")
        return cls(**{name: value[name] for name in fields})

@dataclass(frozen=True)
class SourceChampionizationRecord:
    intake_record_digest: str
    source_evaluation_protocol_digest: str
    selection_receipt_index_digest: str
    attestation_receipt_index_digest: str
    champions: Mapping[str, SourceChampion]
    provisional_selection_digest: str | None = None
    attestation_plan_digest: str | None = None
    competence_mode: str = "OBSERVE"
    selection_rule: str = SELECTION_RULE
    championization_digest: str | None = None
    schema: str = SOURCE_CHAMPIONIZATION_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != SOURCE_CHAMPIONIZATION_SCHEMA
            or self.competence_mode != "OBSERVE"
            or self.selection_rule != SELECTION_RULE
        ):
            raise SourceMarketError("championization schema/mode/rule drifted")
        for name in (
            "intake_record_digest",
            "source_evaluation_protocol_digest",
            "selection_receipt_index_digest",
            "attestation_receipt_index_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (self.provisional_selection_digest is None) != (self.attestation_plan_digest is None):
            raise SourceMarketError("championization formal-stage digests must be both present or absent")
        if self.provisional_selection_digest is not None:
            object.__setattr__(
                self,
                "provisional_selection_digest",
                _digest(self.provisional_selection_digest, "provisional_selection_digest"),
            )
            object.__setattr__(
                self,
                "attestation_plan_digest",
                _digest(self.attestation_plan_digest, "attestation_plan_digest"),
            )
        champions = dict(self.champions)
        if len(champions) != EXPECTED_ANCHOR_COUNT or set(champions) != {
            champion.source_anchor_id for champion in champions.values()
        }:
            raise SourceMarketError("championization must contain exactly one champion per 30 anchors")
        if any(not isinstance(champion, SourceChampion) for champion in champions.values()):
            raise SourceMarketError("championization contains untyped champions")
        if len({champion.candidate_id for champion in champions.values()}) != EXPECTED_ANCHOR_COUNT:
            raise SourceMarketError("one candidate cannot champion multiple anchors")
        object.__setattr__(self, "champions", MappingProxyType(dict(sorted(champions.items()))))
        expected = sha256_json(self._payload_without_digest())
        if self.championization_digest is None:
            object.__setattr__(self, "championization_digest", expected)
        elif _digest(self.championization_digest, "championization_digest") != expected:
            raise SourceMarketError("championization digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intake_record_digest": self.intake_record_digest,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "selection_receipt_index_digest": self.selection_receipt_index_digest,
            "attestation_receipt_index_digest": self.attestation_receipt_index_digest,
            "provisional_selection_digest": self.provisional_selection_digest,
            "attestation_plan_digest": self.attestation_plan_digest,
            "competence_mode": self.competence_mode,
            "selection_rule": self.selection_rule,
            "champions": {
                anchor: champion.to_dict() for anchor, champion in self.champions.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "championization_digest": self.championization_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceChampionizationRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceChampionizationRecord")
        champions = value["champions"]
        if not isinstance(champions, Mapping):
            raise SourceMarketError("SourceChampionizationRecord.champions must be a mapping")
        return cls(
            **{
                name: (
                    {
                        anchor: SourceChampion.from_dict(champion)
                        for anchor, champion in champions.items()
                    }
                    if name == "champions"
                    else value[name]
                )
                for name in fields
            }
        )


def _index_receipts(receipts: Sequence[EvaluatorSourceReceipt], block: EvaluationBlock) -> dict[str, EvaluatorSourceReceipt]:
    result: dict[str, EvaluatorSourceReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, EvaluatorSourceReceipt):
            raise SourceMarketError("source evidence must contain typed evaluator receipts")
        if receipt.block != block:
            raise SourceMarketError(f"{block} evidence contains a receipt from another block")
        if receipt.candidate_id in result:
            raise SourceMarketError(f"duplicate {block} receipt for one candidate")
        result[receipt.candidate_id] = receipt
    return result


def _bind_receipt(
    receipt: EvaluatorSourceReceipt,
    cell: PoolIntakeCell,
    protocol: SourceEvaluationProtocol,
) -> None:
    expected_namespace = (
        protocol.selection_seed_namespace_digest
        if receipt.block == "source_selection"
        else protocol.attestation_seed_namespace_digest
    )
    expected_count = (
        protocol.selection_episodes_per_candidate
        if receipt.block == "source_selection"
        else protocol.attestation_episodes_per_champion
    )
    expected_seeds = (
        protocol.selection_reset_seeds
        if receipt.block == "source_selection"
        else protocol.attestation_reset_seeds
    )
    expected_dataset_digest = sha256_json(
        {
            "schema": SOURCE_ROLLOUT_DATASET_SCHEMA,
            "work_unit_digest": receipt.work_unit_digest,
            "attempt_record_digest": receipt.attempt_record_digest,
            "validated_binding_digest": receipt.validated_binding_digest,
            "evaluation_attempt_number": receipt.evaluation_attempt_number,
            "raw_episode_shard_digest": receipt.raw_episode_shard_digest,
        }
    )
    if (
        receipt.source_evaluation_protocol_digest != protocol.source_evaluation_protocol_digest
        or receipt.intake_record_digest != protocol.intake_record_digest
        or receipt.intake_cell_digest != cell.intake_cell_digest
        or receipt.candidate_id != cell.job_id
        or receipt.source_anchor_id != cell.source_anchor_id
        or receipt.bundle_digest != cell.bundle_digest
        or receipt.source_environment_digest
        != protocol.source_environment_digests[cell.source_anchor_id]
        or receipt.evaluator_implementation_digest != protocol.evaluator_implementation_digest
        or receipt.return_contract_digest != protocol.return_contract_digest
        or receipt.seed_namespace_digest != expected_namespace
        or receipt.episode_count != expected_count
        or receipt.reset_seeds != expected_seeds
        or receipt.dataset_digest != expected_dataset_digest
    ):
        raise SourceMarketError("source evaluator receipt differs from intake/protocol binding")


def _bind_work_unit(
    unit: SourceEvaluationWorkUnit,
    cell: PoolIntakeCell,
    protocol: SourceEvaluationProtocol,
    block: EvaluationBlock,
) -> None:
    supplied_manifest_path = Path(unit.anchor_manifest_path)
    if not supplied_manifest_path.is_absolute() or supplied_manifest_path.is_symlink():
        raise SourceMarketError("source work-unit anchor path is not immutable/absolute")
    try:
        manifest_path = supplied_manifest_path.resolve(strict=True)
    except OSError as error:
        raise SourceMarketError("source work-unit anchor manifest is missing") from error
    manifest = _strict_json_file(manifest_path, "source work-unit anchor manifest")
    if (
        manifest.get("schema") != "policy-learnware.v02-anchor-manifest.v0"
        or manifest.get("manifest_digest") != unit.anchor_manifest_digest
        or unit.anchor_manifest_digest
        != sha256_json(
            {name: item for name, item in manifest.items() if name != "manifest_digest"}
        )
        or not isinstance(manifest.get("runtime"), Mapping)
        or manifest.get("runtime_digest") != unit.anchor_runtime_digest
        or unit.anchor_runtime_digest != sha256_json(manifest["runtime"])
        or manifest.get("anchor_id") != unit.source_anchor_id
        or manifest.get("environment_instance_digest") != unit.source_environment_digest
    ):
        raise SourceMarketError("source work-unit anchor manifest bytes differ from its binding")
    expected_namespace = (
        protocol.selection_seed_namespace_digest
        if block == "source_selection"
        else protocol.attestation_seed_namespace_digest
    )
    expected_seeds = (
        protocol.selection_reset_seeds
        if block == "source_selection"
        else protocol.attestation_reset_seeds
    )
    if (
        not isinstance(unit, SourceEvaluationWorkUnit)
        or unit.block != block
        or unit.source_evaluation_protocol_digest != protocol.source_evaluation_protocol_digest
        or unit.intake_record_digest != protocol.intake_record_digest
        or unit.intake_cell_digest != cell.intake_cell_digest
        or unit.seed_namespace_digest != expected_namespace
        or unit.reset_seeds != expected_seeds
        or unit.candidate_id != cell.job_id
        or unit.source_anchor_id != cell.source_anchor_id
        or unit.attempt_number != cell.attempt_number
        or unit.attempt_digest != cell.attempt_digest
        or unit.bundle_digest != cell.bundle_digest
        or unit.bundle_path != cell.bundle_path
        or unit.outer_iteration != cell.outer_iteration
        or unit.environment_steps != cell.environment_steps
        or unit.source_environment_digest
        != protocol.source_environment_digests[cell.source_anchor_id]
        or unit.evaluator_implementation_digest != protocol.evaluator_implementation_digest
        or unit.return_contract_digest != protocol.return_contract_digest
    ):
        raise SourceMarketError("source evaluation work unit differs from intake/protocol binding")


def provisionally_select_source_pool(
    intake: V03PoolIntakeRecord,
    protocol: SourceEvaluationProtocol,
    selection_work_units: Mapping[str, SourceEvaluationWorkUnit],
    selection_receipts: Sequence[EvaluatorSourceReceipt],
) -> ProvisionalSourceSelection:
    """Freeze 30 winners before any source-attestation outcome exists."""

    if not isinstance(intake, V03PoolIntakeRecord) or intake.pool_state != "POOL_READY":
        raise SourceMarketError("provisional selection requires typed POOL_READY intake")
    if not isinstance(protocol, SourceEvaluationProtocol):
        raise SourceMarketError("provisional selection requires a typed source protocol")
    if protocol.intake_record_digest != intake.intake_record_digest:
        raise SourceMarketError("provisional-selection protocol belongs to another intake")
    units = dict(selection_work_units)
    if len(units) != EXPECTED_JOB_COUNT or set(units) != set(intake.cells):
        raise SourceMarketError("source-selection work units must cover exact-90 candidates")
    receipts = _index_receipts(selection_receipts, "source_selection")
    if not receipts or not set(receipts).issubset(intake.cells):
        raise SourceMarketError("selection receipts contain no usable pool candidates")
    for candidate, receipt in receipts.items():
        cell = intake.cells[candidate]
        _bind_work_unit(units[candidate], cell, protocol, "source_selection")
        _bind_receipt(receipt, cell, protocol)
        if (
            receipt.work_unit_digest != units[candidate].work_unit_digest
            or receipt.runtime_digest
            != units[candidate].anchor_runtime_digest
        ):
            raise SourceMarketError("selection receipt belongs to another source work unit")
    if len({receipt.dataset_digest for receipt in receipts.values()}) != len(receipts):
        raise SourceMarketError("each candidate selection receipt requires its own dataset")
    selected: dict[str, PoolIntakeCell] = {}
    for anchor, candidates in intake.candidates_by_anchor.items():
        rows = [
            (cell, receipts[cell.job_id])
            for cell in candidates
            if cell.job_id in receipts
        ]
        if not rows:
            raise SourceMarketError(
                f"source anchor {anchor} has no candidate that completed a real rollout"
            )
        best_mean = max(receipt.mean for _, receipt in rows)
        tied = [
            (cell, receipt)
            for cell, receipt in rows
            if best_mean - receipt.mean <= protocol.mean_tolerance
        ]
        tied.sort(key=lambda item: (item[1].std, item[0].bundle_digest, item[0].job_id))
        selected[anchor] = tied[0][0]
    selection_index_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-source-selection-receipt-index.v0",
            "receipts": {
                candidate: receipt.receipt_digest
                for candidate, receipt in sorted(receipts.items())
            },
        }
    )
    return ProvisionalSourceSelection(
        intake_record_digest=intake.intake_record_digest,
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        selection_receipt_index_digest=selection_index_digest,
        selected_candidate_ids={anchor: cell.job_id for anchor, cell in selected.items()},
        selected_receipt_digests={
            anchor: receipts[cell.job_id].receipt_digest for anchor, cell in selected.items()
        },
        selected_work_unit_digests={
            anchor: units[cell.job_id].work_unit_digest for anchor, cell in selected.items()
        },
    )


def championize_from_selection(
    intake: V03PoolIntakeRecord,
    protocol: SourceEvaluationProtocol,
    provisional: ProvisionalSourceSelection,
    selection_receipts: Sequence[EvaluatorSourceReceipt],
) -> SourceChampionizationRecord:
    """Publish one champion per anchor from the real selection rollouts.

    v0.3 no longer repeats the same policy evaluation through a second
    admission-only attestation pass.  The selection rollout is the runtime
    evidence: it must load, execute and produce finite normalized returns.
    Competence floors remain reported observations and never veto a runnable
    policy.
    """

    if not isinstance(provisional, ProvisionalSourceSelection):
        raise SourceMarketError("selection championization requires a typed selection")
    if (
        provisional.intake_record_digest != intake.intake_record_digest
        or provisional.source_evaluation_protocol_digest
        != protocol.source_evaluation_protocol_digest
    ):
        raise SourceMarketError("selection belongs to another intake/protocol")
    receipts = _index_receipts(selection_receipts, "source_selection")
    selected_ids = set(provisional.selected_candidate_ids.values())
    if not selected_ids.issubset(receipts):
        raise SourceMarketError("selected candidates lack successful real rollouts")

    champions: dict[str, SourceChampion] = {}
    for anchor, candidate in provisional.selected_candidate_ids.items():
        cell = intake.cells[candidate]
        receipt = receipts[candidate]
        lcb = float(
            receipt.mean
            - protocol.lcb_z * receipt.std / math.sqrt(receipt.episode_count)
        )
        normalized = max(0.0, min(1.0, lcb))
        floor = protocol.competence_floors[anchor]
        competence = SourceCompetenceObservation(
            source_anchor_id=anchor,
            candidate_id=candidate,
            attestation_receipt_digest=receipt.receipt_digest,
            episode_count=receipt.episode_count,
            mean=receipt.mean,
            std=receipt.std,
            lcb=lcb,
            normalized_competence=normalized,
            competence_floor=floor,
            passed=normalized >= floor,
        )
        champions[anchor] = SourceChampion(
            source_anchor_id=anchor,
            candidate_id=candidate,
            seed=cell.seed,
            intake_cell_digest=cell.intake_cell_digest,
            bundle_digest=cell.bundle_digest,
            bundle_path=cell.bundle_path,
            outer_iteration=cell.outer_iteration,
            environment_steps=cell.environment_steps,
            selection_receipt_digest=receipt.receipt_digest,
            # Kept as a backward-compatible field name.  It now points to the
            # same single-pass runtime receipt rather than a duplicated run.
            attestation_receipt_digest=receipt.receipt_digest,
            competence=competence,
        )
    return SourceChampionizationRecord(
        intake_record_digest=intake.intake_record_digest,
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        selection_receipt_index_digest=provisional.selection_receipt_index_digest,
        attestation_receipt_index_digest=provisional.selection_receipt_index_digest,
        champions=champions,
    )

@dataclass(frozen=True)
class V03DeploymentPrivateEntry:
    opaque_learnware_id: str
    candidate_id: str
    source_anchor_id: str
    seed: int
    bundle_digest: str
    bundle_path: str
    outer_iteration: int
    environment_steps: int
    intake_cell_digest: str
    selection_receipt_digest: str
    attestation_receipt_digest: str
    competence_observation_digest: str
    champion_digest: str
    execution_abi: ExecutionABIRecord
    schema: str = DEPLOYMENT_PRIVATE_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DEPLOYMENT_PRIVATE_ENTRY_SCHEMA:
            raise SourceMarketError("unsupported deployment-private schema")
        if not _OPAQUE_ID.fullmatch(_nonempty(self.opaque_learnware_id, "opaque_learnware_id")):
            raise SourceMarketError("deployment opaque ID is invalid")
        if not _CANDIDATE_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceMarketError("deployment candidate ID is invalid")
        object.__setattr__(self, "source_anchor_id", _digest(self.source_anchor_id, "source_anchor_id"))
        for name in (
            "bundle_digest",
            "intake_cell_digest",
            "selection_receipt_digest",
            "attestation_receipt_digest",
            "competence_observation_digest",
            "champion_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.seed not in {0, 1, 2}:
            raise SourceMarketError("deployment seed must be 0/1/2")
        _nonempty(self.bundle_path, "bundle_path")
        object.__setattr__(self, "outer_iteration", _positive_int(self.outer_iteration, "outer_iteration"))
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        if not isinstance(self.execution_abi, ExecutionABIRecord):
            raise SourceMarketError("deployment entry requires a typed private ExecutionABIRecord")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_learnware_id": self.opaque_learnware_id,
            "candidate_id": self.candidate_id,
            "source_anchor_id": self.source_anchor_id,
            "seed": self.seed,
            "bundle_digest": self.bundle_digest,
            "bundle_path": self.bundle_path,
            "outer_iteration": self.outer_iteration,
            "environment_steps": self.environment_steps,
            "intake_cell_digest": self.intake_cell_digest,
            "selection_receipt_digest": self.selection_receipt_digest,
            "attestation_receipt_digest": self.attestation_receipt_digest,
            "competence_observation_digest": self.competence_observation_digest,
            "champion_digest": self.champion_digest,
            "execution_abi": self.execution_abi.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V03DeploymentPrivateEntry":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "V03DeploymentPrivateEntry")
        try:
            abi = ExecutionABIRecord.from_dict(value["execution_abi"])
        except (TypeError, ValueError) as error:
            raise SourceMarketError("deployment entry has an invalid execution ABI") from error
        return cls(
            **{
                name: abi if name == "execution_abi" else value[name]
                for name in fields
            }
        )


@dataclass(frozen=True)
class V03SourcePolicyMarket:
    policy_market_id: str
    intake_record_digest: str
    championization_digest: str
    entries: Mapping[str, PublicMarketEntry]
    deployment_private: Mapping[str, V03DeploymentPrivateEntry]
    anchor_to_opaque_learnware_id: Mapping[str, str]
    asset_state: Literal["ENGINEERING_CONTRACT_ONLY"] = ENGINEERING_MARKET_STATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_market_id", _digest(self.policy_market_id, "policy_market_id"))
        object.__setattr__(self, "intake_record_digest", _digest(self.intake_record_digest, "intake_record_digest"))
        object.__setattr__(self, "championization_digest", _digest(self.championization_digest, "championization_digest"))
        if self.asset_state != ENGINEERING_MARKET_STATE:
            raise SourceMarketError(
                "the current v0.3 market contract has engineering authority only"
            )
        entries = dict(self.entries)
        private = dict(self.deployment_private)
        anchor_map = dict(self.anchor_to_opaque_learnware_id)
        if (
            len(entries) != EXPECTED_ANCHOR_COUNT
            or set(entries) != set(private)
            or set(entries) != set(anchor_map.values())
            or len(anchor_map) != EXPECTED_ANCHOR_COUNT
        ):
            raise SourceMarketError("market public/private/anchor coverage must be exactly 30")
        if any(not isinstance(entry, PublicMarketEntry) for entry in entries.values()):
            raise SourceMarketError("public market contains untyped entries")
        if any(not isinstance(entry, V03DeploymentPrivateEntry) for entry in private.values()):
            raise SourceMarketError("private market contains untyped entries")
        if set(anchor_map) != {entry.source_anchor_id for entry in private.values()}:
            raise SourceMarketError("market anchor index differs from private source anchors")
        for anchor, opaque_id in anchor_map.items():
            _digest(anchor, "anchor_to_opaque_learnware_id key")
            if private[opaque_id].source_anchor_id != anchor:
                raise SourceMarketError("market anchor index points to another source anchor")
        for opaque_id, entry in entries.items():
            if opaque_id != entry.opaque_learnware_id or set(entry.to_dict()) != PUBLIC_ENTRY_ALLOWLIST:
                raise SourceMarketError("public market entry violates the anonymity allowlist")
            if opaque_id not in private or private[opaque_id].opaque_learnware_id != opaque_id:
                raise SourceMarketError("public/private opaque identity mismatch")
        if len({entry.tie_break_token for entry in entries.values()}) != EXPECTED_ANCHOR_COUNT:
            raise SourceMarketError("public tie-break tokens must be unique")
        expected = _market_id(
            self.intake_record_digest,
            self.championization_digest,
            entries,
            private,
        )
        if self.policy_market_id != expected:
            raise SourceMarketError("policy_market_id does not match public/private bindings")
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(entries.items()))))
        object.__setattr__(self, "deployment_private", MappingProxyType(dict(sorted(private.items()))))
        object.__setattr__(
            self,
            "anchor_to_opaque_learnware_id",
            MappingProxyType(dict(sorted(anchor_map.items()))),
        )

    def public_manifest(self) -> dict[str, Any]:
        result = {
            "schema": PUBLIC_MARKET_SCHEMA,
            "policy_market_id": self.policy_market_id,
            "entries": {
                opaque_id: entry.to_dict() for opaque_id, entry in self.entries.items()
            },
        }
        if set(result) != PUBLIC_MANIFEST_ALLOWLIST:
            raise SourceMarketError("public manifest violates the anonymity allowlist")
        return result

    def deployment_manifest(self) -> dict[str, Any]:
        return {
            "schema": PRIVATE_MARKET_SCHEMA,
            "policy_market_id": self.policy_market_id,
            "intake_record_digest": self.intake_record_digest,
            "championization_digest": self.championization_digest,
            "entries": {
                opaque_id: entry.to_dict()
                for opaque_id, entry in self.deployment_private.items()
            },
            "anchor_to_opaque_learnware_id": dict(
                self.anchor_to_opaque_learnware_id
            ),
        }

    @classmethod
    def from_manifests(
        cls,
        public_manifest: Mapping[str, Any],
        deployment_manifest: Mapping[str, Any],
    ) -> "V03SourcePolicyMarket":
        _strict(public_manifest, set(PUBLIC_MANIFEST_ALLOWLIST), "public source market")
        _strict(
            deployment_manifest,
            {
                "schema",
                "policy_market_id",
                "intake_record_digest",
                "championization_digest",
                "entries",
                "anchor_to_opaque_learnware_id",
            },
            "deployment-private source market",
        )
        if public_manifest["schema"] != PUBLIC_MARKET_SCHEMA:
            raise SourceMarketError("unsupported public source-market schema")
        if deployment_manifest["schema"] != PRIVATE_MARKET_SCHEMA:
            raise SourceMarketError("unsupported deployment-private source-market schema")
        if public_manifest["policy_market_id"] != deployment_manifest["policy_market_id"]:
            raise SourceMarketError("public/private policy_market_id mismatch")
        public_entries = public_manifest["entries"]
        private_entries = deployment_manifest["entries"]
        anchor_map = deployment_manifest["anchor_to_opaque_learnware_id"]
        if not isinstance(public_entries, Mapping) or not isinstance(private_entries, Mapping):
            raise SourceMarketError("source-market entries must be mappings")
        if not isinstance(anchor_map, Mapping):
            raise SourceMarketError("source-market anchor index must be a mapping")
        try:
            public = {
                opaque_id: PublicMarketEntry.from_dict(entry)
                for opaque_id, entry in public_entries.items()
            }
        except (TypeError, ValueError) as error:
            raise SourceMarketError("public source-market entry is invalid") from error
        private = {
            opaque_id: V03DeploymentPrivateEntry.from_dict(entry)
            for opaque_id, entry in private_entries.items()
        }
        return cls(
            policy_market_id=deployment_manifest["policy_market_id"],
            intake_record_digest=deployment_manifest["intake_record_digest"],
            championization_digest=deployment_manifest["championization_digest"],
            entries=public,
            deployment_private=private,
            anchor_to_opaque_learnware_id=anchor_map,
        )


def _deployment_binding_digest(
    private: Mapping[str, V03DeploymentPrivateEntry],
) -> str:
    return sha256_json(
        {
            opaque_id: entry.to_dict()
            for opaque_id, entry in sorted(private.items())
        }
    )


def _market_id(
    intake_record_digest: str,
    championization_digest: str,
    public: Mapping[str, PublicMarketEntry],
    private: Mapping[str, V03DeploymentPrivateEntry],
) -> str:
    return sha256_json(
        {
            "schema": MARKET_ID_SCHEMA,
            "intake_record_digest": intake_record_digest,
            "championization_digest": championization_digest,
            "entries": {
                opaque_id: entry.to_dict() for opaque_id, entry in sorted(public.items())
            },
            "deployment_binding_digest": _deployment_binding_digest(private),
        }
    )


def market_nonce_commitment(
    *,
    purpose: Literal["market_alias", "market_tie_break"],
    nonce: str,
    intake_record_digest: str,
) -> str:
    """Commit a private nonce without publishing the nonce itself."""

    if purpose not in {"market_alias", "market_tie_break"}:
        raise SourceMarketError("unsupported private market nonce purpose")
    return sha256_json(
        {
            "schema": PRIVATE_NONCE_COMMITMENT_SCHEMA,
            "purpose": purpose,
            "intake_record_digest": _digest(
                intake_record_digest, "intake_record_digest"
            ),
            "private_nonce": _nonce(nonce, f"{purpose}_nonce"),
        }
    )


def formal_market_alias_protocol_digest(
    *,
    intake_record_digest: str,
    source_pool_digest: str,
    alias_commitment_digest: str,
    candidate_count: int = EXPECTED_JOB_COUNT,
    market_entry_count: int = EXPECTED_ANCHOR_COUNT,
) -> str:
    """Recompute the frozen alias protocol commitment used by the formal plan."""

    if (
        candidate_count != EXPECTED_JOB_COUNT
        or market_entry_count != EXPECTED_ANCHOR_COUNT
    ):
        raise SourceMarketError("formal alias protocol requires exact 90-to-30 counts")
    return sha256_json(
        {
            "schema": MARKET_ALIAS_PROTOCOL_SCHEMA,
            "intake_record_digest": _digest(
                intake_record_digest, "intake_record_digest"
            ),
            "source_pool_digest": _digest(source_pool_digest, "source_pool_digest"),
            "candidate_count": candidate_count,
            "market_entry_count": market_entry_count,
            "assignment": MARKET_ALIAS_ASSIGNMENT,
            "alias_commitment_digest": _digest(
                alias_commitment_digest, "alias_commitment_digest"
            ),
        }
    )


def derive_market_opaque_id(*, candidate_id: str, market_alias_nonce: str) -> str:
    """Derive the public alias for one champion under the private alias nonce."""

    candidate = _nonempty(candidate_id, "candidate_id")
    nonce = _nonce(market_alias_nonce, "market_alias_nonce")
    return "lw-" + sha256_json(
        {
            "schema": MARKET_ALIAS_SCHEMA,
            "market_alias_nonce": nonce,
            "candidate_id": candidate,
        }
    )[:32]


def derive_market_tie_break_token(
    *, candidate_id: str, tie_break_nonce: str
) -> str:
    """Derive the public deterministic tie token for one champion."""

    candidate = _nonempty(candidate_id, "candidate_id")
    nonce = _nonce(tie_break_nonce, "tie_break_nonce")
    return sha256_json(
        {
            "schema": MARKET_TIE_BREAK_SCHEMA,
            "tie_break_nonce": nonce,
            "candidate_id": candidate,
        }
    )


def build_source_policy_market(
    championization: SourceChampionizationRecord,
    execution_abis: Mapping[str, ExecutionABIRecord],
    *,
    market_alias_nonce: str,
    tie_break_nonce: str,
) -> V03SourcePolicyMarket:
    """Publish one source champion per anchor with an anonymous public view."""

    if not isinstance(championization, SourceChampionizationRecord):
        raise SourceMarketError("market construction requires typed championization")
    alias_nonce = _nonce(market_alias_nonce, "market_alias_nonce")
    tie_nonce = _nonce(tie_break_nonce, "tie_break_nonce")
    if alias_nonce == tie_nonce:
        raise SourceMarketError("market alias and tie-break domains require distinct nonces")
    candidate_ids = {champion.candidate_id for champion in championization.champions.values()}
    if set(execution_abis) != candidate_ids or any(
        not isinstance(abi, ExecutionABIRecord) for abi in execution_abis.values()
    ):
        raise SourceMarketError("execution ABI registry must cover exactly the 30 champions")
    public: dict[str, PublicMarketEntry] = {}
    private: dict[str, V03DeploymentPrivateEntry] = {}
    anchor_map: dict[str, str] = {}
    for anchor, champion in championization.champions.items():
        opaque_id = derive_market_opaque_id(
            candidate_id=champion.candidate_id,
            market_alias_nonce=alias_nonce,
        )
        if opaque_id in public:
            raise SourceMarketError("market alias collision")
        token = derive_market_tie_break_token(
            candidate_id=champion.candidate_id,
            tie_break_nonce=tie_nonce,
        )
        public[opaque_id] = PublicMarketEntry(
            opaque_learnware_id=opaque_id,
            normalized_source_competence=champion.competence.normalized_competence,
            tie_break_token=token,
        )
        private[opaque_id] = V03DeploymentPrivateEntry(
            opaque_learnware_id=opaque_id,
            candidate_id=champion.candidate_id,
            source_anchor_id=anchor,
            seed=champion.seed,
            bundle_digest=champion.bundle_digest,
            bundle_path=champion.bundle_path,
            outer_iteration=champion.outer_iteration,
            environment_steps=champion.environment_steps,
            intake_cell_digest=champion.intake_cell_digest,
            selection_receipt_digest=champion.selection_receipt_digest,
            attestation_receipt_digest=champion.attestation_receipt_digest,
            competence_observation_digest=champion.competence.observation_digest,
            champion_digest=champion.champion_digest,
            execution_abi=execution_abis[champion.candidate_id],
        )
        anchor_map[anchor] = opaque_id
    policy_market_id = _market_id(
        championization.intake_record_digest,
        championization.championization_digest,
        public,
        private,
    )
    return V03SourcePolicyMarket(
        policy_market_id=policy_market_id,
        intake_record_digest=championization.intake_record_digest,
        championization_digest=championization.championization_digest,
        entries=public,
        deployment_private=private,
        anchor_to_opaque_learnware_id=anchor_map,
    )


__all__ = [
    "ENGINEERING_MARKET_STATE",
    "MARKET_ALIAS_ASSIGNMENT",
    "MARKET_ALIAS_PROTOCOL_SCHEMA",
    "MARKET_ALIAS_SCHEMA",
    "MARKET_TIE_BREAK_SCHEMA",
    "PRIVATE_NONCE_COMMITMENT_SCHEMA",
    "EvaluatorSourceReceipt",
    "ProvisionalSourceSelection",
    "RawSourceEpisodeShard",
    "PUBLIC_ENTRY_ALLOWLIST",
    "PUBLIC_MANIFEST_ALLOWLIST",
    "SELECTION_RULE",
    "SourceChampion",
    "SourceChampionizationRecord",
    "SourceCompetenceObservation",
    "SourceEvaluationProtocol",
    "SourceEvaluationWorkUnit",
    "SourceMarketError",
    "V03DeploymentPrivateEntry",
    "V03SourcePolicyMarket",
    "build_source_policy_market",
    "championize_from_selection",
    "derive_market_opaque_id",
    "derive_market_tie_break_token",
    "formal_market_alias_protocol_digest",
    "build_source_evaluation_work_unit",
    "market_nonce_commitment",
    "provisionally_select_source_pool",
    "receipt_from_source_episode_shard",
]
