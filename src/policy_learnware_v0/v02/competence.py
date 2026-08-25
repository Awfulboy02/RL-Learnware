"""Source-only seed/checkpoint championization and independent attestation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .config import V02ExperimentConfig
from .schemas import SourceCompetenceRecord
from .training import (
    AdmittedTrainingRecord,
    admitted_training_records_digest,
    validate_admitted_training_grid,
)


EvaluationBlock = Literal["source_selection", "source_attestation"]


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise ValueError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 digest") from exc
    return result


@dataclass(frozen=True)
class SourceEpisodeRow:
    source_anchor_id: str
    candidate_id: str
    bundle_digest: str
    block: EvaluationBlock
    reset_seed: int
    normalized_return: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_anchor_id", _digest(self.source_anchor_id, "source_anchor_id")
        )
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "bundle_digest", _digest(self.bundle_digest, "bundle_digest"))
        if self.block not in {"source_selection", "source_attestation"}:
            raise ValueError("invalid source evaluation block")
        if isinstance(self.reset_seed, bool) or not isinstance(self.reset_seed, int) or self.reset_seed < 0:
            raise ValueError("source evaluation seed must be non-negative")
        value = float(self.normalized_return)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("DMC normalized source return must lie in [0, 1]")
        object.__setattr__(self, "normalized_return", value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CandidateSelectionSummary:
    source_anchor_id: str
    candidate_id: str
    bundle_digest: str
    episode_count: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_anchor_id", _digest(self.source_anchor_id, "source_anchor_id")
        )
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "bundle_digest", _digest(self.bundle_digest, "bundle_digest"))
        if isinstance(self.episode_count, bool) or not isinstance(self.episode_count, int) or self.episode_count <= 0:
            raise ValueError("selection summary episode_count must be positive")
        for name in ("mean", "std"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"selection summary {name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.mean <= 1.0 or self.std < 0.0:
            raise ValueError("selection summary statistics are outside their valid range")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FormalChampionizationAdmission:
    """Immutable binding from reviewed config and admitted jobs to evaluation."""

    config_digest: str
    admitted_records_digest: str
    expected_anchor_ids: tuple[str, ...]
    expected_candidate_ids: tuple[str, ...]
    selection_episodes_per_candidate: int
    attestation_episodes_per_champion: int
    competence_floors_digest: str
    championization_protocol_digest: str
    selection_digest: str
    private_attestation_index_digest: str

    def __post_init__(self) -> None:
        for name in (
            "config_digest",
            "admitted_records_digest",
            "competence_floors_digest",
            "championization_protocol_digest",
            "selection_digest",
            "private_attestation_index_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        anchors = tuple(sorted(_digest(item, "expected_anchor_ids[]") for item in self.expected_anchor_ids))
        candidates = tuple(sorted(_nonempty(item, "expected_candidate_ids[]") for item in self.expected_candidate_ids))
        if not anchors or len(anchors) != len(set(anchors)):
            raise ValueError("formal expected anchor IDs must be non-empty and unique")
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("formal expected candidate IDs must be non-empty and unique")
        object.__setattr__(self, "expected_anchor_ids", anchors)
        object.__setattr__(self, "expected_candidate_ids", candidates)
        for name in (
            "selection_episodes_per_candidate",
            "attestation_episodes_per_champion",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-formal-championization-admission.v0",
            "config_digest": self.config_digest,
            "admitted_records_digest": self.admitted_records_digest,
            "expected_anchor_ids": list(self.expected_anchor_ids),
            "expected_candidate_ids": list(self.expected_candidate_ids),
            "selection_episodes_per_candidate": self.selection_episodes_per_candidate,
            "attestation_episodes_per_champion": self.attestation_episodes_per_champion,
            "competence_floors_digest": self.competence_floors_digest,
            "championization_protocol_digest": self.championization_protocol_digest,
            "selection_digest": self.selection_digest,
            "private_attestation_index_digest": self.private_attestation_index_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalChampionizationAdmission":
        if not isinstance(value, Mapping):
            raise ValueError("formal championization admission must be a mapping")
        fields = set(cls.__dataclass_fields__)
        expected = fields | {"schema"}
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "invalid formal championization admission keys; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if value["schema"] != "policy-learnware.v02-formal-championization-admission.v0":
            raise ValueError("unsupported formal championization admission schema")
        return cls(
            **{
                name: (
                    tuple(value[name])
                    if name in {"expected_anchor_ids", "expected_candidate_ids"}
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class ChampionizationResult:
    selected_by_anchor: Mapping[str, str]
    selection_summaries: tuple[CandidateSelectionSummary, ...]
    competence_records: Mapping[str, SourceCompetenceRecord]
    rejected_anchors: Mapping[str, str]
    selection_digest: str
    attested_bundle_digests: Mapping[str, str] = field(default_factory=dict)
    formal_admission: FormalChampionizationAdmission | None = None

    def __post_init__(self) -> None:
        selected = {
            _digest(anchor, "selected source anchor"): _nonempty(candidate, "selected candidate")
            for anchor, candidate in self.selected_by_anchor.items()
        }
        if not selected or len(set(selected.values())) != len(selected):
            raise ValueError("championization must select one distinct candidate per anchor")
        summaries = tuple(self.selection_summaries)
        if not summaries or any(not isinstance(item, CandidateSelectionSummary) for item in summaries):
            raise ValueError("championization requires typed selection summaries")
        summary_pairs = {(item.source_anchor_id, item.candidate_id) for item in summaries}
        if len(summary_pairs) != len(summaries):
            raise ValueError("championization contains duplicate candidate summaries")
        if {item.source_anchor_id for item in summaries} != set(selected):
            raise ValueError("selection summaries and selected anchors differ")
        selected_summaries = {
            anchor: tuple(
                item for item in summaries
                if item.source_anchor_id == anchor and item.candidate_id == candidate
            )
            for anchor, candidate in selected.items()
        }
        if any(len(items) != 1 for items in selected_summaries.values()):
            raise ValueError("selected candidate lacks exactly one selection summary")
        competence = dict(self.competence_records)
        rejected = dict(self.rejected_anchors)
        if set(competence) & set(rejected) or set(competence) | set(rejected) != set(selected):
            raise ValueError("competence and rejection outcomes must partition selected anchors")
        for anchor, record in competence.items():
            if not isinstance(record, SourceCompetenceRecord):
                raise ValueError("competence records must be typed")
            if record.opaque_source_anchor_id != anchor:
                raise ValueError("competence record belongs to another source anchor")
            if record.championization_digest != self.selection_digest:
                raise ValueError("competence record is bound to another selection")
        attested = {
            _digest(anchor, "attested source anchor"): _digest(bundle, "attested bundle digest")
            for anchor, bundle in self.attested_bundle_digests.items()
        }
        if attested:
            if set(attested) != set(selected):
                raise ValueError("attested bundle digests must cover every selected anchor")
            for anchor, items in selected_summaries.items():
                if attested[anchor] != items[0].bundle_digest:
                    raise ValueError("selection and attestation bundle digests differ")
        if self.formal_admission is not None:
            if not isinstance(self.formal_admission, FormalChampionizationAdmission):
                raise ValueError("formal admission must be typed")
            if set(self.formal_admission.expected_anchor_ids) != set(selected):
                raise ValueError("formal admission and selected source-anchor sets differ")
            if set(self.formal_admission.expected_candidate_ids) != {
                item.candidate_id for item in summaries
            }:
                raise ValueError("formal admission and evaluated candidate sets differ")
            if self.formal_admission.selection_digest != self.selection_digest:
                raise ValueError("formal admission is bound to another selection")
            if not attested:
                raise ValueError("formal championization requires attested bundle bindings")
            if not rejected:
                expected_floors_digest = sha256_json(
                    {
                        anchor: record.competence_floor
                        for anchor, record in sorted(competence.items())
                    }
                )
                if self.formal_admission.competence_floors_digest != expected_floors_digest:
                    raise ValueError("formal admission competence floors differ from results")
                expected_attestation_index = sha256_json(
                    {
                        anchor: {
                            "bundle_digest": attested[anchor],
                            "private_attestation_digest": competence[anchor].private_attestation_digest,
                        }
                        for anchor in sorted(selected)
                    }
                )
                if (
                    self.formal_admission.private_attestation_index_digest
                    != expected_attestation_index
                ):
                    raise ValueError("formal admission is bound to other attestation evidence")
        object.__setattr__(self, "selection_digest", _digest(self.selection_digest, "selection_digest"))
        object.__setattr__(self, "selected_by_anchor", MappingProxyType(dict(sorted(selected.items()))))
        object.__setattr__(self, "selection_summaries", summaries)
        object.__setattr__(self, "competence_records", MappingProxyType(dict(sorted(competence.items()))))
        object.__setattr__(self, "rejected_anchors", MappingProxyType(dict(sorted(rejected.items()))))
        object.__setattr__(self, "attested_bundle_digests", MappingProxyType(dict(sorted(attested.items()))))

    @property
    def selected_bundle_digests(self) -> Mapping[str, str]:
        selected = dict(self.selected_by_anchor)
        return MappingProxyType(
            {
                anchor: next(
                    item.bundle_digest
                    for item in self.selection_summaries
                    if item.source_anchor_id == anchor and item.candidate_id == candidate
                )
                for anchor, candidate in selected.items()
            }
        )


def _summary(rows: Sequence[SourceEpisodeRow]) -> CandidateSelectionSummary:
    if not rows:
        raise ValueError("cannot summarize empty source evaluation rows")
    identity = {(row.source_anchor_id, row.candidate_id, row.bundle_digest) for row in rows}
    if len(identity) != 1:
        raise ValueError("source evaluation summary rows have mixed identities")
    seeds = [row.reset_seed for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("source evaluation contains duplicate episode seeds")
    values = np.asarray([row.normalized_return for row in rows], dtype=np.float64)
    anchor, candidate, bundle = next(iter(identity))
    return CandidateSelectionSummary(
        source_anchor_id=anchor,
        candidate_id=candidate,
        bundle_digest=bundle,
        episode_count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=0)),
    )


def championize_by_anchor(
    selection_rows: Sequence[SourceEpisodeRow],
    attestation_rows: Sequence[SourceEpisodeRow],
    *,
    competence_floors: Mapping[str, float],
    mean_tolerance: float,
    lcb_z: float | None,
    return_contract_id: str,
) -> ChampionizationResult:
    """Select on one seed block, publish competence from another, never fall back."""

    if not math.isfinite(float(mean_tolerance)) or mean_tolerance < 0.0:
        raise ValueError("mean_tolerance must be finite and non-negative")
    if lcb_z is not None and (not math.isfinite(float(lcb_z)) or lcb_z < 0.0):
        raise ValueError("lcb_z must be finite and non-negative")
    if any(row.block != "source_selection" for row in selection_rows):
        raise ValueError("selection_rows must come only from source_selection")
    if any(row.block != "source_attestation" for row in attestation_rows):
        raise ValueError("attestation_rows must come only from source_attestation")

    selection_seeds = {row.reset_seed for row in selection_rows}
    attestation_seeds = {row.reset_seed for row in attestation_rows}
    overlap = selection_seeds & attestation_seeds
    if overlap:
        raise ValueError(f"source selection and attestation seeds overlap: {sorted(overlap)}")
    grouped: dict[tuple[str, str], list[SourceEpisodeRow]] = {}
    for row in selection_rows:
        grouped.setdefault((row.source_anchor_id, row.candidate_id), []).append(row)
    summaries = tuple(
        _summary(rows) for _, rows in sorted(grouped.items(), key=lambda item: item[0])
    )
    by_anchor: dict[str, list[CandidateSelectionSummary]] = {}
    for summary in summaries:
        by_anchor.setdefault(summary.source_anchor_id, []).append(summary)
    if set(by_anchor) != set(competence_floors):
        raise ValueError("competence floors must cover exactly the evaluated source anchors")

    selected: dict[str, str] = {}
    selected_summaries: dict[str, CandidateSelectionSummary] = {}
    for anchor, candidates in sorted(by_anchor.items()):
        best_mean = max(item.mean for item in candidates)
        tied = [item for item in candidates if best_mean - item.mean <= float(mean_tolerance)]
        tied.sort(key=lambda item: (item.std, item.bundle_digest, item.candidate_id))
        winner = tied[0]
        selected[anchor] = winner.candidate_id
        selected_summaries[anchor] = winner

    selection_payload = {
        "schema": "policy-learnware.v02-championization.v0",
        "mean_tolerance": float(mean_tolerance),
        "rows": [item.to_dict() for item in summaries],
        "selected": dict(sorted(selected.items())),
    }
    selection_digest = sha256_json(selection_payload)

    attested_groups: dict[tuple[str, str], list[SourceEpisodeRow]] = {}
    for row in attestation_rows:
        if row.source_anchor_id not in selected:
            raise ValueError("attestation row references an unevaluated anchor")
        if row.candidate_id != selected[row.source_anchor_id]:
            raise ValueError("attestation may evaluate only the already selected champion")
        attested_groups.setdefault((row.source_anchor_id, row.candidate_id), []).append(row)
    if {anchor for anchor, _ in attested_groups} != set(selected):
        raise ValueError("every selected champion requires independent attestation rows")

    competence: dict[str, SourceCompetenceRecord] = {}
    rejected: dict[str, str] = {}
    for (anchor, candidate), rows in sorted(attested_groups.items()):
        summary = _summary(rows)
        lcb = (
            None
            if lcb_z is None
            else float(summary.mean - float(lcb_z) * summary.std / math.sqrt(summary.episode_count))
        )
        normalized = summary.mean if lcb is None else max(0.0, min(1.0, lcb))
        floor = float(competence_floors[anchor])
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError("competence floors must lie in [0, 1]")
        attestation_digest = sha256_json(
            {
                "schema": "policy-learnware.v02-private-source-attestation.v0",
                "rows": [row.to_dict() for row in rows],
            }
        )
        seed_digest = sha256_json(
            {
                "namespace": "source_attestation",
                "seeds": sorted(row.reset_seed for row in rows),
            }
        )
        learnware_id = "v02l-" + sha256_json(
            {"anchor": anchor, "candidate": candidate, "selection": selection_digest}
        )[:24]
        record = SourceCompetenceRecord(
            learnware_id=learnware_id,
            opaque_source_anchor_id=anchor,
            return_contract_id=return_contract_id,
            validation_seed_digest=seed_digest,
            episode_count=summary.episode_count,
            mean=summary.mean,
            std=summary.std,
            lcb=lcb,
            normalized_competence=normalized,
            competence_floor=floor,
            passed=normalized >= floor,
            championization_digest=selection_digest,
            private_attestation_digest=attestation_digest,
        )
        if record.passed:
            competence[anchor] = record
        else:
            rejected[anchor] = "selected_champion_failed_independent_attestation"
    return ChampionizationResult(
        selected_by_anchor=selected,
        selection_summaries=summaries,
        competence_records=competence,
        rejected_anchors=rejected,
        selection_digest=selection_digest,
        attested_bundle_digests={
            anchor: _summary(rows).bundle_digest
            for (anchor, _), rows in sorted(attested_groups.items())
        },
    )


def admit_formal_championization(
    config: V02ExperimentConfig,
    admitted_records: Mapping[str, AdmittedTrainingRecord],
    selection_rows: Sequence[SourceEpisodeRow],
    attestation_rows: Sequence[SourceEpisodeRow],
    *,
    mean_tolerance: float,
    lcb_z: float | None,
    return_contract_id: str,
) -> ChampionizationResult:
    """Strictly admit source championization against the reviewed v0.2 freeze.

    Scientific choices not represented by :class:`V02ExperimentConfig` remain
    explicit required inputs.  This function never supplies values for them;
    it only binds them into the admission protocol digest.
    """

    if not isinstance(config, V02ExperimentConfig):
        raise ValueError("formal championization requires a typed v0.2 config")
    if config.stage != "v02_freeze_ready":
        raise ValueError("formal championization requires stage v02_freeze_ready")
    expected_anchors = config.source_anchor_ids
    candidates_by_anchor = validate_admitted_training_grid(
        admitted_records,
        expected_anchor_ids=expected_anchors,
        expected_seeds=config.training_seeds,
        algorithm=config.primary_algorithm,
        environment_steps=config.training_steps,
        checkpoint_rule=config.checkpoint_rule,
    )
    for candidate_id, record in admitted_records.items():
        if record.job.config_digest != config.config_digest:
            raise ValueError(
                f"formal admitted candidate {candidate_id!r} is bound to another config"
            )
        if record.job.execution_purpose != "v02_freeze_ready":
            raise ValueError(
                f"formal admitted candidate {candidate_id!r} has non-formal execution purpose"
            )
        if not record.attestation.is_server_bound:
            raise ValueError(
                f"formal admitted candidate {candidate_id!r} lacks raw server provenance"
            )
    admitted_digest = admitted_training_records_digest(admitted_records)
    expected_candidates = tuple(sorted(admitted_records))

    selection = tuple(selection_rows)
    attestation = tuple(attestation_rows)
    for row in selection + attestation:
        if not isinstance(row, SourceEpisodeRow):
            raise ValueError("formal source evidence must contain typed episode rows")
        record = admitted_records.get(row.candidate_id)
        if record is None:
            raise ValueError("source episode row references an unadmitted candidate")
        if row.source_anchor_id != record.job.source_anchor_id:
            raise ValueError("source episode row assigns an admitted candidate to another anchor")
        if row.bundle_digest != record.attestation.bundle_digest:
            raise ValueError("source episode bundle digest differs from admitted training")

    selection_groups: dict[str, list[SourceEpisodeRow]] = {}
    for row in selection:
        selection_groups.setdefault(row.candidate_id, []).append(row)
    if set(selection_groups) != set(expected_candidates):
        raise ValueError("selection evidence must cover exactly every admitted candidate")
    if any(
        len(rows) != config.source_eval_episodes.selection_episodes
        for rows in selection_groups.values()
    ):
        raise ValueError("selection episode count differs from reviewed config")
    selection_seed_banks = {
        tuple(sorted(row.reset_seed for row in rows)) for rows in selection_groups.values()
    }
    if len(selection_seed_banks) != 1:
        raise ValueError("all admitted candidates must use the same source-selection seed bank")
    for anchor, candidates in candidates_by_anchor.items():
        if {row.candidate_id for row in selection if row.source_anchor_id == anchor} != set(candidates):
            raise ValueError("selection candidates differ from the admitted anchor/seed grid")

    floors = config.source_competence_floor_by_anchor
    result = championize_by_anchor(
        selection,
        attestation,
        competence_floors=floors,
        mean_tolerance=mean_tolerance,
        lcb_z=lcb_z,
        return_contract_id=return_contract_id,
    )
    attestation_groups: dict[str, list[SourceEpisodeRow]] = {}
    for row in attestation:
        attestation_groups.setdefault(row.candidate_id, []).append(row)
    if set(attestation_groups) != set(result.selected_by_anchor.values()):
        raise ValueError("attestation evidence must cover exactly the selected champions")
    if any(
        len(rows) != config.source_eval_episodes.attestation_episodes
        for rows in attestation_groups.values()
    ):
        raise ValueError("attestation episode count differs from reviewed config")
    attestation_seed_banks = {
        tuple(sorted(row.reset_seed for row in rows)) for rows in attestation_groups.values()
    }
    if len(attestation_seed_banks) != 1:
        raise ValueError("all champions must use the same source-attestation seed bank")

    return_digest = _digest(return_contract_id, "return_contract_id")
    floors_digest = sha256_json(dict(sorted(floors.items())))
    protocol_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-championization-protocol.v0",
            "mean_tolerance": float(mean_tolerance),
            "lcb_z": None if lcb_z is None else float(lcb_z),
            "return_contract_id": return_digest,
            "selection_episodes_per_candidate": config.source_eval_episodes.selection_episodes,
            "attestation_episodes_per_champion": config.source_eval_episodes.attestation_episodes,
        }
    )
    admission = FormalChampionizationAdmission(
        config_digest=config.config_digest,
        admitted_records_digest=admitted_digest,
        expected_anchor_ids=expected_anchors,
        expected_candidate_ids=expected_candidates,
        selection_episodes_per_candidate=config.source_eval_episodes.selection_episodes,
        attestation_episodes_per_champion=config.source_eval_episodes.attestation_episodes,
        competence_floors_digest=floors_digest,
        championization_protocol_digest=protocol_digest,
        selection_digest=result.selection_digest,
        private_attestation_index_digest=sha256_json(
            {
                anchor: {
                    "bundle_digest": result.attested_bundle_digests[anchor],
                    "private_attestation_digest": (
                        result.competence_records[anchor].private_attestation_digest
                        if anchor in result.competence_records
                        else sha256_json(
                            {
                                "schema": "policy-learnware.v02-private-source-attestation.v0",
                                "rows": [
                                    row.to_dict()
                                    for row in attestation
                                    if row.source_anchor_id == anchor
                                ],
                            }
                        )
                    ),
                }
                for anchor in sorted(result.selected_by_anchor)
            }
        ),
    )
    return replace(result, formal_admission=admission)


__all__ = [
    "CandidateSelectionSummary",
    "ChampionizationResult",
    "EvaluationBlock",
    "FormalChampionizationAdmission",
    "SourceEpisodeRow",
    "admit_formal_championization",
    "championize_by_anchor",
]
