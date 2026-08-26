"""Paired legacy attribution replay and G03-Attribution evidence records.

This module deliberately owns no historical writer.  A caller supplies an
immutable :class:`TransitionBank` and a replay adapter bound to an archived
checkpoint.  All reports are new v0.3 records and preserve the raw dataset and
view digests needed for independent recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import canonicalize, sha256_json
from .transition_views import (
    REGISTERED_VIEW_IDS,
    TRANSITION_VIEW_PROTOCOL_ID,
    VIEW_REGISTRY,
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_SHUFFLED_NEXT,
    V_SHUFFLED_REWARD,
    TransitionBank,
    TransitionViewResult,
    apply_transition_view,
)


FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
ATTRIBUTION_PREFIX_SCHEDULE_SCHEMA = (
    "policy-learnware.v03-attribution-prefix-schedule.v0"
)

ATTRIBUTION_REPLAY_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-attribution-replay-protocol.v0",
        "view_protocol_id": TRANSITION_VIEW_PROTOCOL_ID,
        "comparison": "paired-against-full-legacy",
        "formal_prefix_episode_counts": list(
            FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS
        ),
        "historical_artifact_mutation": "forbidden",
        "causal_claim": "nested-and-destructive-controls-only",
    }
)

REQUIRED_ATTRIBUTION_VIEW_IDS = REGISTERED_VIEW_IDS


class AttributionError(ValueError):
    """Attribution evidence is incomplete, non-finite, or not reproducible."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AttributionError(f"{name} must be a non-empty canonical string")
    return value


def _digest(value: Any, name: str) -> str:
    result = _nonempty(value, name)
    if len(result) != 64:
        raise AttributionError(f"{name} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise AttributionError(f"{name} must be a SHA-256 digest") from error
    if result != result.lower():
        raise AttributionError(f"{name} must use lowercase hexadecimal")
    return result


def _finite_mapping(value: Mapping[str, float], name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise AttributionError(f"{name} must be a non-empty mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        _nonempty(key, f"{name} key")
        if isinstance(raw, bool) or not isinstance(
            raw, (int, float, np.integer, np.floating)
        ):
            raise AttributionError(f"{name}.{key} must be numeric")
        number = float(raw)
        if not np.isfinite(number):
            raise AttributionError(f"{name}.{key} must be finite")
        result[key] = number
    return MappingProxyType(dict(sorted(result.items())))


def _prefix_curves(
    value: Mapping[str, Mapping[int, float]],
) -> Mapping[str, Mapping[int, float]]:
    if not isinstance(value, Mapping) or not value:
        raise AttributionError("prefix_curves must be a non-empty mapping")
    result: dict[str, Mapping[int, float]] = {}
    for metric, curve in value.items():
        _nonempty(metric, "prefix metric")
        if not isinstance(curve, Mapping) or not curve:
            raise AttributionError(f"prefix curve {metric} cannot be empty")
        points: dict[int, float] = {}
        for prefix, raw in curve.items():
            if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix <= 0:
                raise AttributionError("prefix keys must be positive integers")
            if isinstance(raw, bool) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                raise AttributionError(f"prefix curve {metric}[{prefix}] is not numeric")
            number = float(raw)
            if not np.isfinite(number):
                raise AttributionError(f"prefix curve {metric}[{prefix}] is non-finite")
            points[prefix] = number
        if tuple(points) != tuple(sorted(points)):
            points = dict(sorted(points.items()))
        result[metric] = MappingProxyType(points)
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class AttributionPrefixSchedule:
    """Digest-bound prefix grid for archived attribution replay.

    Development fixtures may use a strict subset so CPU tests remain small,
    but only the exact preregistered 1/2/4/8/16/32/64 grid is formal-eligible.
    The scope is persisted in every report; a short smoke cannot masquerade as
    the archived replay protocol.
    """

    prefix_episode_counts: tuple[int, ...]
    scope: Literal["FORMAL", "DEVELOPMENT"]
    schema: str = ATTRIBUTION_PREFIX_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ATTRIBUTION_PREFIX_SCHEDULE_SCHEMA:
            raise AttributionError("unsupported attribution prefix schedule schema")
        prefixes = tuple(self.prefix_episode_counts)
        if (
            not prefixes
            or any(type(value) is not int or value <= 0 for value in prefixes)
            or prefixes != tuple(sorted(set(prefixes)))
        ):
            raise AttributionError(
                "attribution prefixes must be positive, unique, and increasing"
            )
        if self.scope not in {"FORMAL", "DEVELOPMENT"}:
            raise AttributionError("unknown attribution prefix schedule scope")
        if (
            self.scope == "FORMAL"
            and prefixes != FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS
        ):
            raise AttributionError(
                "formal attribution requires prefixes 1/2/4/8/16/32/64"
            )
        object.__setattr__(self, "prefix_episode_counts", prefixes)

    @classmethod
    def formal(cls) -> "AttributionPrefixSchedule":
        return cls(FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS, "FORMAL")

    @classmethod
    def development(
        cls, prefix_episode_counts: Sequence[int]
    ) -> "AttributionPrefixSchedule":
        return cls(tuple(prefix_episode_counts), "DEVELOPMENT")

    @property
    def formal_eligible(self) -> bool:
        return self.scope == "FORMAL"

    @property
    def schedule_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prefix_episode_counts": list(self.prefix_episode_counts),
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
        }


@dataclass(frozen=True)
class AttributionMeasurement:
    """Adapter output in a shared metric space for one transition view."""

    view_id: str
    task_group: str
    shared_schema_group: str
    retrieval_metrics: Mapping[str, float]
    between_within_mmd_summaries: Mapping[str, float]
    prefix_curves: Mapping[str, Mapping[int, float]]
    failure_identifiability_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.view_id not in VIEW_REGISTRY:
            raise AttributionError(f"unregistered view: {self.view_id!r}")
        _nonempty(self.task_group, "task_group")
        _nonempty(self.shared_schema_group, "shared_schema_group")
        object.__setattr__(
            self,
            "retrieval_metrics",
            _finite_mapping(self.retrieval_metrics, "retrieval_metrics"),
        )
        object.__setattr__(
            self,
            "between_within_mmd_summaries",
            _finite_mapping(
                self.between_within_mmd_summaries,
                "between_within_mmd_summaries",
            ),
        )
        object.__setattr__(self, "prefix_curves", _prefix_curves(self.prefix_curves))
        notes = tuple(_nonempty(note, "failure/identifiability note") for note in self.failure_identifiability_notes)
        if not notes:
            raise AttributionError("failure_identifiability_notes cannot be empty")
        object.__setattr__(self, "failure_identifiability_notes", notes)

    @property
    def flat_metrics(self) -> Mapping[str, float]:
        result = {
            **{f"retrieval.{key}": value for key, value in self.retrieval_metrics.items()},
            **{
                f"mmd.{key}": value
                for key, value in self.between_within_mmd_summaries.items()
            },
        }
        for metric, curve in self.prefix_curves.items():
            for prefix, value in curve.items():
                result[f"prefix.{metric}@{prefix}"] = value
        return MappingProxyType(dict(sorted(result.items())))


@runtime_checkable
class LegacyReplayAdapter(Protocol):
    """Narrow adapter around a frozen legacy encoder/RKME replay."""

    encoder_checkpoint_digest: str
    implementation_digest: str

    def replay(
        self,
        view: TransitionViewResult,
        *,
        prefix_episode_counts: tuple[int, ...],
    ) -> AttributionMeasurement: ...


@dataclass(frozen=True)
class CallableLegacyReplayAdapter:
    """Concrete bridge for the existing v0/v0.1 replay implementation.

    The callable is execution-only and therefore excluded from persisted
    records.  Its implementation digest must be supplied from the frozen
    legacy runtime/code manifest rather than inferred from a Python repr.
    """

    encoder_checkpoint_digest: str
    implementation_digest: str
    replay_callable: Callable[
        [TransitionViewResult, tuple[int, ...]], AttributionMeasurement
    ]

    def __post_init__(self) -> None:
        _digest(self.encoder_checkpoint_digest, "encoder_checkpoint_digest")
        _digest(self.implementation_digest, "implementation_digest")
        if not callable(self.replay_callable):
            raise AttributionError("replay_callable must be callable")

    def replay(
        self,
        view: TransitionViewResult,
        *,
        prefix_episode_counts: tuple[int, ...],
    ) -> AttributionMeasurement:
        measurement = self.replay_callable(view, prefix_episode_counts)
        if not isinstance(measurement, AttributionMeasurement):
            raise AttributionError("legacy replay callable returned an invalid record")
        return measurement


@dataclass(frozen=True)
class ArchivedLegacyReference:
    archive_protocol_id: str
    archive_manifest_digest: str
    archived_dataset_digest: str
    canonical_bank_digest: str
    encoder_checkpoint_digest: str
    encoder_implementation_digest: str
    reference_metrics: Mapping[str, float]
    absolute_tolerance: float
    relative_tolerance: float

    def __post_init__(self) -> None:
        _nonempty(self.archive_protocol_id, "archive_protocol_id")
        for name in (
            "archive_manifest_digest",
            "archived_dataset_digest",
            "canonical_bank_digest",
            "encoder_checkpoint_digest",
            "encoder_implementation_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "reference_metrics",
            _finite_mapping(self.reference_metrics, "reference_metrics"),
        )
        for name in ("absolute_tolerance", "relative_tolerance"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise AttributionError(f"{name} must be numeric")
            number = float(raw)
            if not np.isfinite(number) or number < 0:
                raise AttributionError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, number)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "archive_protocol_id": self.archive_protocol_id,
                "archive_manifest_digest": self.archive_manifest_digest,
                "archived_dataset_digest": self.archived_dataset_digest,
                "canonical_bank_digest": self.canonical_bank_digest,
                "encoder_checkpoint_digest": self.encoder_checkpoint_digest,
                "encoder_implementation_digest": self.encoder_implementation_digest,
                "reference_metrics": dict(self.reference_metrics),
                "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance,
            }
        )


@dataclass(frozen=True)
class AttributionReport:
    view_protocol_id: str
    view_id: str
    input_channel_allowlist: tuple[str, ...]
    encoder_checkpoint_digest: str
    encoder_implementation_digest: str
    archived_dataset_digest: str
    canonical_bank_digest: str
    transition_view_digest: str
    prefix_schedule_digest: str
    prefix_schedule_scope: Literal["FORMAL", "DEVELOPMENT"]
    task_group: str
    shared_schema_group: str
    retrieval_metrics: Mapping[str, float]
    between_within_mmd_summaries: Mapping[str, float]
    prefix_curves: Mapping[str, Mapping[int, float]]
    paired_deltas_vs_full_legacy: Mapping[str, float]
    shuffled_control_deltas: Mapping[str, float]
    failure_identifiability_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.view_protocol_id != TRANSITION_VIEW_PROTOCOL_ID:
            raise AttributionError("AttributionReport uses an unknown view protocol")
        if self.view_id not in VIEW_REGISTRY:
            raise AttributionError("AttributionReport uses an unregistered view")
        if self.input_channel_allowlist != VIEW_REGISTRY[self.view_id].input_channel_allowlist:
            raise AttributionError("input channel allowlist disagrees with registry")
        for name in (
            "encoder_checkpoint_digest",
            "encoder_implementation_digest",
            "archived_dataset_digest",
            "canonical_bank_digest",
            "transition_view_digest",
            "prefix_schedule_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _nonempty(self.task_group, "task_group")
        _nonempty(self.shared_schema_group, "shared_schema_group")
        if self.prefix_schedule_scope not in {"FORMAL", "DEVELOPMENT"}:
            raise AttributionError("unknown attribution prefix schedule scope")
        object.__setattr__(
            self,
            "retrieval_metrics",
            _finite_mapping(self.retrieval_metrics, "retrieval_metrics"),
        )
        object.__setattr__(
            self,
            "between_within_mmd_summaries",
            _finite_mapping(
                self.between_within_mmd_summaries,
                "between_within_mmd_summaries",
            ),
        )
        object.__setattr__(self, "prefix_curves", _prefix_curves(self.prefix_curves))
        object.__setattr__(
            self,
            "paired_deltas_vs_full_legacy",
            _finite_mapping(
                self.paired_deltas_vs_full_legacy,
                "paired_deltas_vs_full_legacy",
            ),
        )
        object.__setattr__(
            self,
            "shuffled_control_deltas",
            _finite_mapping(
                self.shuffled_control_deltas,
                "shuffled_control_deltas",
            ),
        )
        notes = tuple(
            _nonempty(note, "failure/identifiability note")
            for note in self.failure_identifiability_notes
        )
        if not notes:
            raise AttributionError("failure_identifiability_notes cannot be empty")
        object.__setattr__(self, "failure_identifiability_notes", notes)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-attribution-report.v0",
            "attribution_replay_protocol_id": ATTRIBUTION_REPLAY_PROTOCOL_ID,
            "view_protocol_id": self.view_protocol_id,
            "view_id": self.view_id,
            "input_channel_allowlist": list(self.input_channel_allowlist),
            "encoder_checkpoint_digest": self.encoder_checkpoint_digest,
            "encoder_implementation_digest": self.encoder_implementation_digest,
            "archived_dataset_digest": self.archived_dataset_digest,
            "canonical_bank_digest": self.canonical_bank_digest,
            "transition_view_digest": self.transition_view_digest,
            "prefix_schedule_digest": self.prefix_schedule_digest,
            "prefix_schedule_scope": self.prefix_schedule_scope,
            "formal_prefix_schedule": self.prefix_schedule_scope == "FORMAL",
            "task_group": self.task_group,
            "shared_schema_group": self.shared_schema_group,
            "retrieval_metrics": dict(self.retrieval_metrics),
            "between_within_mmd_summaries": dict(self.between_within_mmd_summaries),
            "prefix_curves": {
                metric: {str(prefix): value for prefix, value in curve.items()}
                for metric, curve in self.prefix_curves.items()
            },
            "paired_deltas_vs_full_legacy": dict(self.paired_deltas_vs_full_legacy),
            "shuffled_control_deltas": dict(self.shuffled_control_deltas),
            "failure_identifiability_notes": list(self.failure_identifiability_notes),
        }


# The foundation sidecar has no formal gate authority.  A future archived-data
# recompute loader must define a separate, authority-bound record before formal
# PASS can exist; keeping PASS constructible here would let callers self-attest.
AttributionGateStatus = Literal["DEVELOPMENT_PASS", "FAIL"]
DynamicsInterpretation = Literal[
    "LEGACY_ENCODER_DYNAMICS_SENSITIVE",
    "LEGACY_ENCODER_NOT_DYNAMICS_SENSITIVE",
    "UNASSESSED",
]


@dataclass(frozen=True)
class AttributionGateEvidence:
    gate_status: AttributionGateStatus
    evidence_scope: Literal["SYNTHETIC", "LEGACY_ARCHIVED"]
    full_legacy_replay_pass: bool
    controls_fail_closed_pass: bool
    contribution_quantified_pass: bool
    shared_schema_explanation_pass: bool
    independently_recomputable_pass: bool
    dynamics_interpretation: DynamicsInterpretation
    maximum_legacy_replay_error: float
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gate_status == "PASS":
            raise AttributionError(
                "formal attribution PASS is unavailable until archived replay "
                "and independent-recompute authority are implemented"
            )
        if self.gate_status not in {"DEVELOPMENT_PASS", "FAIL"}:
            raise AttributionError("unknown attribution gate status")
        if self.evidence_scope not in {"SYNTHETIC", "LEGACY_ARCHIVED"}:
            raise AttributionError("unknown attribution evidence scope")
        if self.dynamics_interpretation not in {
            "LEGACY_ENCODER_DYNAMICS_SENSITIVE",
            "LEGACY_ENCODER_NOT_DYNAMICS_SENSITIVE",
            "UNASSESSED",
        }:
            raise AttributionError("unknown dynamics interpretation")
        if (
            isinstance(self.maximum_legacy_replay_error, bool)
            or not isinstance(self.maximum_legacy_replay_error, (int, float))
            or self.maximum_legacy_replay_error < 0
            or not np.isfinite(self.maximum_legacy_replay_error)
        ):
            raise AttributionError("maximum legacy replay error is invalid")
        reasons = tuple(_nonempty(reason, "failure reason") for reason in self.failure_reasons)
        core_flags = (
            self.full_legacy_replay_pass,
            self.controls_fail_closed_pass,
            self.contribution_quantified_pass,
            self.shared_schema_explanation_pass,
        )
        if any(type(value) is not bool for value in (*core_flags, self.independently_recomputable_pass)):
            raise AttributionError("attribution gate flags must be booleans")
        if self.gate_status == "DEVELOPMENT_PASS":
            if self.evidence_scope != "SYNTHETIC" or not all(core_flags):
                raise AttributionError(
                    "development attribution PASS requires synthetic scope and core checks"
                )
            if reasons:
                raise AttributionError("development attribution PASS cannot carry failure reasons")
        elif not reasons:
            raise AttributionError("failed attribution gate requires failure reasons")
        object.__setattr__(self, "failure_reasons", reasons)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-attribution-gate-evidence.v0",
                **canonicalize(self.__dict__),
            }
        )


@dataclass(frozen=True)
class AttributionSuite:
    reports: tuple[AttributionReport, ...]
    archived_reference_digest: str
    prefix_schedule: AttributionPrefixSchedule
    gate_evidence: AttributionGateEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.prefix_schedule, AttributionPrefixSchedule):
            raise AttributionError("suite requires a typed prefix schedule")
        if any(
            report.prefix_schedule_digest
            != self.prefix_schedule.schedule_digest
            or report.prefix_schedule_scope != self.prefix_schedule.scope
            for report in self.reports
        ):
            raise AttributionError("reports disagree with the suite prefix schedule")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-attribution-suite.v0",
                "report_digests": [report.digest for report in self.reports],
                "archived_reference_digest": self.archived_reference_digest,
                "prefix_schedule": self.prefix_schedule.to_dict(),
                "prefix_schedule_digest": self.prefix_schedule.schedule_digest,
                "gate_evidence_digest": self.gate_evidence.digest,
            }
        )


def _paired_delta(
    measurement: AttributionMeasurement,
    full: AttributionMeasurement,
) -> Mapping[str, float]:
    current = measurement.flat_metrics
    baseline = full.flat_metrics
    common = sorted(set(current) & set(baseline))
    if not common:
        raise AttributionError("view and full legacy replay share no metrics")
    return MappingProxyType({name: current[name] - baseline[name] for name in common})


def _replay_error(
    observed: Mapping[str, float], reference: ArchivedLegacyReference
) -> tuple[bool, float, tuple[str, ...]]:
    missing = sorted(set(reference.reference_metrics) - set(observed))
    if missing:
        return False, 0.0, tuple(f"MISSING_ARCHIVED_METRIC:{name}" for name in missing)
    maximum = 0.0
    failures: list[str] = []
    for name, expected in reference.reference_metrics.items():
        actual = observed[name]
        error = abs(actual - expected)
        maximum = max(maximum, error)
        tolerance = reference.absolute_tolerance + reference.relative_tolerance * abs(expected)
        if error > tolerance:
            failures.append(f"LEGACY_REPLAY_MISMATCH:{name}")
    return not failures, maximum, tuple(failures)


def run_attribution_replay(
    bank: TransitionBank,
    adapter: LegacyReplayAdapter,
    archived_reference: ArchivedLegacyReference,
    *,
    prefix_episode_counts: Sequence[int] | None = None,
    prefix_schedule: AttributionPrefixSchedule | None = None,
    view_ids: Sequence[str] = REQUIRED_ATTRIBUTION_VIEW_IDS,
    shuffle_seed: int = 0,
    dynamics_sensitivity_tolerance: float = 1.0e-8,
) -> AttributionSuite:
    """Execute paired views and emit development evidence.

    This runner deliberately cannot mint a formal ``PASS``.  Formal archived
    evidence additionally needs a separately admitted archive authority and an
    independent recompute receipt, neither of which can be replaced by a
    caller-supplied scope string.
    """

    if not isinstance(bank, TransitionBank):
        raise AttributionError("bank must be a TransitionBank")
    if (
        isinstance(dynamics_sensitivity_tolerance, bool)
        or not isinstance(dynamics_sensitivity_tolerance, (int, float))
        or not np.isfinite(dynamics_sensitivity_tolerance)
        or dynamics_sensitivity_tolerance < 0
    ):
        raise AttributionError(
            "dynamics_sensitivity_tolerance must be finite and nonnegative"
        )
    checkpoint_digest = _digest(
        adapter.encoder_checkpoint_digest, "adapter.encoder_checkpoint_digest"
    )
    implementation_digest = _digest(
        adapter.implementation_digest, "adapter.implementation_digest"
    )
    if checkpoint_digest != archived_reference.encoder_checkpoint_digest:
        raise AttributionError("adapter checkpoint does not match archived reference")
    if implementation_digest != archived_reference.encoder_implementation_digest:
        raise AttributionError("adapter implementation does not match archived reference")
    if bank.archived_dataset_digest != archived_reference.archived_dataset_digest:
        raise AttributionError("transition bank does not match archived reference")
    if bank.canonical_bank_digest != archived_reference.canonical_bank_digest:
        raise AttributionError("adapted transition arrays do not match archived reference")
    if (prefix_episode_counts is None) == (prefix_schedule is None):
        raise AttributionError(
            "supply exactly one of prefix_episode_counts or prefix_schedule"
        )
    if prefix_schedule is None:
        # Backward-compatible short fixtures are explicitly development-only.
        schedule = AttributionPrefixSchedule.development(
            tuple(prefix_episode_counts or ())
        )
    else:
        if not isinstance(prefix_schedule, AttributionPrefixSchedule):
            raise AttributionError("prefix_schedule must be typed")
        schedule = prefix_schedule
    prefixes = schedule.prefix_episode_counts
    episode_count = int(bank.episode_offsets.size - 1)
    if (
        not prefixes
        or prefixes != tuple(sorted(set(prefixes)))
        or prefixes[0] <= 0
        or prefixes[-1] > episode_count
    ):
        raise AttributionError("prefix_episode_counts must be unique, sorted, and in range")
    requested = tuple(view_ids)
    if len(set(requested)) != len(requested) or any(
        view_id not in VIEW_REGISTRY for view_id in requested
    ):
        raise AttributionError("view_ids must be unique registered views")
    if V_FULL_LEGACY not in requested:
        raise AttributionError("V_FULL_LEGACY is mandatory for paired attribution")
    views: dict[str, TransitionViewResult] = {}
    measurements: dict[str, AttributionMeasurement] = {}
    for view_id in requested:
        result = apply_transition_view(
            bank,
            view_id,
            shuffle_seed=shuffle_seed,
        )
        measurement = adapter.replay(result, prefix_episode_counts=prefixes)
        if not isinstance(measurement, AttributionMeasurement):
            raise AttributionError("legacy adapter returned an invalid measurement")
        if measurement.view_id != view_id:
            raise AttributionError("legacy adapter returned the wrong view ID")
        if any(tuple(curve) != prefixes for curve in measurement.prefix_curves.values()):
            raise AttributionError(
                "legacy adapter prefix curves disagree with the frozen schedule"
            )
        views[view_id] = result
        measurements[view_id] = measurement
    full = measurements[V_FULL_LEGACY]
    shuffled_deltas: dict[str, float] = {}
    for control_id in (V_SHUFFLED_NEXT, V_SHUFFLED_REWARD):
        if control_id in measurements:
            for name, value in _paired_delta(measurements[control_id], full).items():
                shuffled_deltas[f"{control_id}:{name}"] = value
    if not shuffled_deltas:
        shuffled_deltas["UNAVAILABLE"] = 0.0
    reports = tuple(
        AttributionReport(
            view_protocol_id=TRANSITION_VIEW_PROTOCOL_ID,
            view_id=view_id,
            input_channel_allowlist=VIEW_REGISTRY[view_id].input_channel_allowlist,
            encoder_checkpoint_digest=checkpoint_digest,
            encoder_implementation_digest=implementation_digest,
            archived_dataset_digest=str(bank.archived_dataset_digest),
            canonical_bank_digest=bank.canonical_bank_digest,
            transition_view_digest=views[view_id].view_digest,
            prefix_schedule_digest=schedule.schedule_digest,
            prefix_schedule_scope=schedule.scope,
            task_group=measurements[view_id].task_group,
            shared_schema_group=measurements[view_id].shared_schema_group,
            retrieval_metrics=measurements[view_id].retrieval_metrics,
            between_within_mmd_summaries=measurements[
                view_id
            ].between_within_mmd_summaries,
            prefix_curves=measurements[view_id].prefix_curves,
            paired_deltas_vs_full_legacy=_paired_delta(measurements[view_id], full),
            shuffled_control_deltas=shuffled_deltas,
            failure_identifiability_notes=measurements[
                view_id
            ].failure_identifiability_notes,
        )
        for view_id in requested
    )
    replay_pass, replay_error, replay_reasons = _replay_error(
        full.flat_metrics, archived_reference
    )
    required_present = set(REQUIRED_ATTRIBUTION_VIEW_IDS).issubset(measurements)
    controls_fail_closed = required_present and all(
        set(views[view_id].channels)
        == (
            {"random_embedding"}
            if view_id == V_RANDOM_ENCODER
            else set(VIEW_REGISTRY[view_id].input_channel_allowlist)
        )
        for view_id in REQUIRED_ATTRIBUTION_VIEW_IDS
    )
    contribution_quantified = required_present and all(
        bool(report.paired_deltas_vs_full_legacy) for report in reports
    )
    shared_schema_pass = all(
        measurement.shared_schema_group not in {"", "UNSPECIFIED"}
        and bool(measurement.failure_identifiability_notes)
        for measurement in measurements.values()
    )
    # Digest completeness makes a later recompute possible, but is not itself
    # an independent recomputation.  Only the external formal gate may set this
    # check after admitting a second replay receipt.
    recomputable = False
    reasons = list(replay_reasons)
    if not controls_fail_closed:
        reasons.append("INCOMPLETE_OR_NON_FAIL_CLOSED_CONTROLS")
    if not contribution_quantified:
        reasons.append("CONTRIBUTION_DELTAS_INCOMPLETE")
    if not shared_schema_pass:
        reasons.append("SHARED_SCHEMA_IDENTIFIABILITY_UNEXPLAINED")
    next_deltas = [
        abs(value)
        for name, value in shuffled_deltas.items()
        if name.startswith(V_SHUFFLED_NEXT + ":")
    ]
    if not next_deltas:
        dynamics: DynamicsInterpretation = "UNASSESSED"
    elif max(next_deltas) <= dynamics_sensitivity_tolerance:
        dynamics = "LEGACY_ENCODER_NOT_DYNAMICS_SENSITIVE"
    else:
        dynamics = "LEGACY_ENCODER_DYNAMICS_SENSITIVE"
    if reasons:
        gate_status: AttributionGateStatus = "FAIL"
    else:
        gate_status = "DEVELOPMENT_PASS"
    gate = AttributionGateEvidence(
        gate_status=gate_status,
        evidence_scope="SYNTHETIC",
        full_legacy_replay_pass=replay_pass,
        controls_fail_closed_pass=controls_fail_closed,
        contribution_quantified_pass=contribution_quantified,
        shared_schema_explanation_pass=shared_schema_pass,
        independently_recomputable_pass=recomputable,
        dynamics_interpretation=dynamics,
        maximum_legacy_replay_error=replay_error,
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )
    return AttributionSuite(
        reports=reports,
        archived_reference_digest=archived_reference.digest,
        prefix_schedule=schedule,
        gate_evidence=gate,
    )
