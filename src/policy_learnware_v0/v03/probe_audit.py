"""Coverage, nuisance, cost, and No-Go audits for public probes."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..probe.dataset import EpisodeDataset
from .probes import (
    CP0_EXACT_COMMON,
    CP2_UNSEEN_PROBE,
    FROZEN_PROBE_STYLES,
    ActionABI,
    ProbeCollectionReceipt,
    ProbeContractError,
    ProbeSeedBinding,
    ProbeTrainingManifest,
    assert_candidate_independent,
    registered_probe,
    validate_cp2_holdout,
)
from .representation_ladder import R0_PADDED_RAW


PROBE_AUDIT_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-probe-audit-protocol.v2",
        "saturation_threshold": 0.98,
        "state_coverage": "mean-per-dimension-standard-deviation",
        "raw_signal": "median-l2-next-minus-current",
        "spectrum": "normalized-rfft-spectral-centroid",
        "invariance_ratio": "median-cross-style/median-cross-dynamics",
        "collection_evidence": "frozen-replay-plus-native-action-equality",
        "authority": "development-only-formal-authority-not-implemented",
    }
)


class ProbeAuditError(ValueError):
    """Raw probe evidence or a gate request is malformed."""


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ProbeAuditError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ProbeAuditError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ProbeAuditError(f"{name} must be strictly positive")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise ProbeAuditError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ProbeAuditError(
            f"{name} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _rate(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise ProbeAuditError(f"{name} must lie in [0, 1]")
    return result


def _distances(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(_finite_nonnegative(value, name) for value in values)
    if not result:
        raise ProbeAuditError(f"{name} cannot be empty")
    return result


@dataclass(frozen=True)
class PrefixCost:
    episode_count: int
    wall_seconds: float
    stored_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_count, bool)
            or not isinstance(self.episode_count, int)
            or self.episode_count <= 0
        ):
            raise ProbeAuditError("prefix episode_count must be positive")
        object.__setattr__(
            self, "wall_seconds", _finite_nonnegative(self.wall_seconds, "wall_seconds")
        )
        if (
            isinstance(self.stored_bytes, bool)
            or not isinstance(self.stored_bytes, int)
            or self.stored_bytes < 0
        ):
            raise ProbeAuditError("stored_bytes must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "wall_seconds": self.wall_seconds,
            "stored_bytes": self.stored_bytes,
        }


@dataclass(frozen=True)
class ProbeBankSummary:
    task_id: str
    context_id: str
    probe_style_id: str
    regime: str
    dataset_digest: str
    transition_count: int
    episode_count: int
    finite: bool
    action_saturation_rate: float
    action_energy: float
    action_spectral_centroid: float
    state_displacement: float
    state_coverage: float
    termination_rate: float
    failure_rate: float
    raw_transition_signal: float
    prefix_costs: tuple[PrefixCost, ...]
    collection_receipt: ProbeCollectionReceipt
    semantic_style_hidden: bool

    def __post_init__(self) -> None:
        for name in ("task_id", "context_id", "probe_style_id", "dataset_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ProbeAuditError(f"{name} must be a non-empty canonical string")
        _digest(self.dataset_digest, "dataset_digest")
        if self.probe_style_id not in FROZEN_PROBE_STYLES:
            raise ProbeAuditError("probe style is not registered")
        if FROZEN_PROBE_STYLES[self.probe_style_id].regime != self.regime:
            raise ProbeAuditError("probe regime disagrees with registered style")
        if self.transition_count <= 0 or self.episode_count <= 0:
            raise ProbeAuditError("bank counts must be positive")
        for name in (
            "action_saturation_rate",
            "termination_rate",
            "failure_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name))
        for name in (
            "action_energy",
            "action_spectral_centroid",
            "state_displacement",
            "state_coverage",
            "raw_transition_signal",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if not self.prefix_costs:
            raise ProbeAuditError("prefix_costs cannot be empty")
        counts = tuple(record.episode_count for record in self.prefix_costs)
        if counts != tuple(sorted(set(counts))) or counts[-1] > self.episode_count:
            raise ProbeAuditError("prefix costs must be unique, sorted, and in range")
        if not isinstance(self.collection_receipt, ProbeCollectionReceipt):
            raise ProbeAuditError("collection_receipt must be typed replay evidence")
        receipt = self.collection_receipt
        if (
            receipt.probe_style_id != self.probe_style_id
            or receipt.dataset_digest != self.dataset_digest
            or receipt.transition_count != self.transition_count
            or receipt.episode_count != self.episode_count
        ):
            raise ProbeAuditError(
                "collection receipt disagrees with probe-bank summary"
            )
        if not isinstance(self.semantic_style_hidden, bool):
            raise ProbeAuditError("semantic_style_hidden must be boolean")

    @property
    def candidate_independence_pass(self) -> bool:
        return self.collection_receipt.candidate_independence_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "context_id": self.context_id,
            "probe_style_id": self.probe_style_id,
            "regime": self.regime,
            "dataset_digest": self.dataset_digest,
            "transition_count": self.transition_count,
            "episode_count": self.episode_count,
            "finite": self.finite,
            "action_saturation_rate": self.action_saturation_rate,
            "action_energy": self.action_energy,
            "action_spectral_centroid": self.action_spectral_centroid,
            "state_displacement": self.state_displacement,
            "state_coverage": self.state_coverage,
            "termination_rate": self.termination_rate,
            "failure_rate": self.failure_rate,
            "raw_transition_signal": self.raw_transition_signal,
            "prefix_costs": [record.to_dict() for record in self.prefix_costs],
            "collection_receipt": self.collection_receipt.to_dict(),
            "collection_receipt_digest": self.collection_receipt.digest,
            "candidate_independence_pass": self.candidate_independence_pass,
            "semantic_style_hidden": self.semantic_style_hidden,
        }


def _spectral_centroid(actions: np.ndarray, offsets: np.ndarray) -> float:
    numerators: list[float] = []
    denominators: list[float] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        episode = actions[int(start) : int(stop)]
        if episode.shape[0] < 2:
            continue
        spectrum = np.square(np.abs(np.fft.rfft(episode, axis=0)))
        frequencies = np.fft.rfftfreq(episode.shape[0])[:, None]
        numerators.append(float(np.sum(frequencies * spectrum)))
        denominators.append(float(np.sum(spectrum)))
    denominator = float(np.sum(denominators))
    return 0.0 if denominator <= 0.0 else float(np.sum(numerators)) / denominator


def summarize_probe_bank(
    dataset: EpisodeDataset,
    *,
    action_abi: ActionABI,
    seed_bindings: Sequence[ProbeSeedBinding],
    collection_implementation_digest: str,
    task_id: str,
    context_id: str,
    probe_style_id: str,
    collection_wall_seconds: float,
    stored_bytes: int,
    prefix_episode_counts: Sequence[int] | None = None,
    explicit_failure_count: int = 0,
    semantic_style_hidden: bool = True,
) -> ProbeBankSummary:
    """Replay, verify, receipt, and summarize one raw probe bank.

    ``dataset.action`` is the native action written by the environment
    collector.  The normalized action stream is never trusted as caller
    evidence: it is regenerated from the registered frozen policy and the
    typed seed bindings, then mapped through ``action_abi`` and compared row by
    row with the stored dataset.
    """

    if not isinstance(dataset, EpisodeDataset):
        raise ProbeAuditError("dataset must be an EpisodeDataset")
    if not isinstance(action_abi, ActionABI):
        raise ProbeAuditError("action_abi must be an ActionABI")
    _digest(collection_implementation_digest, "collection_implementation_digest")
    try:
        style = FROZEN_PROBE_STYLES[probe_style_id]
    except KeyError as error:
        raise ProbeAuditError(f"unregistered probe style: {probe_style_id!r}") from error
    try:
        assert_candidate_independent(style)
    except ProbeContractError as error:  # pragma: no cover - frozen registry is valid
        raise ProbeAuditError(str(error)) from error
    bindings = tuple(seed_bindings)
    if len(bindings) != dataset.episode_count or not all(
        isinstance(binding, ProbeSeedBinding) for binding in bindings
    ):
        raise ProbeAuditError(
            "seed_bindings must contain one typed binding per episode"
        )
    if tuple(binding.episode_id for binding in bindings) != tuple(
        range(dataset.episode_count)
    ):
        raise ProbeAuditError("probe seed binding episode IDs must be contiguous")
    if any(binding.style_id != probe_style_id for binding in bindings):
        raise ProbeAuditError("probe seed binding style differs from requested style")
    roles = {binding.role for binding in bindings}
    if len(roles) != 1:
        raise ProbeAuditError("one probe bank cannot mix source and target roles")
    namespaces = {binding.namespace for binding in bindings}
    nonces = {binding.nonce for binding in bindings}
    if len(namespaces) != 1 or len(nonces) != 1:
        raise ProbeAuditError("one probe bank must use one frozen seed namespace")
    expected_seeds = np.asarray(
        [binding.seed for binding in bindings], dtype=np.int64
    )
    if not np.array_equal(dataset.probe_seeds, expected_seeds):
        raise ProbeAuditError("dataset probe seeds differ from frozen bindings")
    if dataset.action_dim != action_abi.action_dim:
        raise ProbeAuditError("dataset action dimension disagrees with Action ABI")

    probe = registered_probe(probe_style_id)
    normalized_rows: list[np.ndarray] = []
    native_rows: list[np.ndarray] = []
    for episode_index, binding in enumerate(bindings):
        state = probe.reset(int(binding.seed), action_abi)
        episode = dataset.episode_slice(episode_index)
        for step, row_index in enumerate(range(episode.start, episode.stop)):
            normalized, state = probe.act(
                dataset.observation[row_index], state, step=step
            )
            normalized_rows.append(normalized)
            native_rows.append(action_abi.map_normalized(normalized))
    actions = np.asarray(np.stack(normalized_rows), dtype=np.float32)
    replayed_native = np.asarray(np.stack(native_rows), dtype=np.float32)
    if not np.array_equal(dataset.action, replayed_native):
        raise ProbeAuditError(
            "dataset native actions differ from frozen probe replay and Action ABI mapping"
        )
    finite = bool(
        np.all(np.isfinite(actions))
        and np.all(np.isfinite(dataset.observation))
        and np.all(np.isfinite(dataset.next_observation))
    )
    if not finite:
        raise ProbeAuditError("probe bank contains non-finite raw values")
    tolerance = 8.0 * np.finfo(np.float32).eps
    if np.any(actions < -1.0 - tolerance) or np.any(actions > 1.0 + tolerance):
        raise ProbeAuditError("replayed normalized actions lie outside [-1, 1]")
    if explicit_failure_count < 0 or explicit_failure_count > dataset.episode_count:
        raise ProbeAuditError("explicit_failure_count lies outside episode count")
    delta = np.asarray(dataset.next_observation, dtype=np.float64) - np.asarray(
        dataset.observation, dtype=np.float64
    )
    ended_terminated = 0
    for stop in dataset.episode_offsets[1:]:
        ended_terminated += int(dataset.terminated[int(stop) - 1])
    episode_counts = (
        tuple(prefix_episode_counts)
        if prefix_episode_counts is not None
        else tuple(
            value
            for value in (1, 2, 4, 8, 16, 32, 64, dataset.episode_count)
            if value <= dataset.episode_count
        )
    )
    episode_counts = tuple(sorted(set(int(value) for value in episode_counts)))
    if not episode_counts or episode_counts[-1] > dataset.episode_count or episode_counts[0] <= 0:
        raise ProbeAuditError("prefix episode counts are invalid")
    total_seconds = _finite_nonnegative(collection_wall_seconds, "collection_wall_seconds")
    if isinstance(stored_bytes, bool) or not isinstance(stored_bytes, int) or stored_bytes < 0:
        raise ProbeAuditError("stored_bytes must be a nonnegative integer")
    prefix_costs = tuple(
        PrefixCost(
            episode_count=count,
            wall_seconds=total_seconds * count / dataset.episode_count,
            stored_bytes=int(np.ceil(stored_bytes * count / dataset.episode_count)),
        )
        for count in episode_counts
    )
    collection_receipt = ProbeCollectionReceipt(
        role=next(iter(roles)),
        probe_style_id=probe_style_id,
        style_digest=style.digest,
        action_abi_digest=action_abi.digest,
        collection_implementation_digest=collection_implementation_digest,
        seed_binding_digests=tuple(binding.digest for binding in bindings),
        seed_sequence_digest=sha256_ndarrays(
            {"probe_seeds": np.asarray(expected_seeds, dtype=np.int64)}
        ),
        dataset_digest=dataset.digest,
        normalized_actions_digest=sha256_ndarrays(
            {"normalized_actions": actions}
        ),
        native_actions_digest=sha256_ndarrays(
            {"native_actions": replayed_native}
        ),
        episode_count=dataset.episode_count,
        transition_count=dataset.transition_count,
    )
    return ProbeBankSummary(
        task_id=task_id,
        context_id=context_id,
        probe_style_id=probe_style_id,
        regime=style.regime,
        dataset_digest=dataset.digest,
        transition_count=dataset.transition_count,
        episode_count=dataset.episode_count,
        finite=finite,
        action_saturation_rate=float(np.mean(np.abs(actions) >= 0.98)),
        action_energy=float(np.mean(np.square(actions))),
        action_spectral_centroid=_spectral_centroid(
            actions, dataset.episode_offsets
        ),
        state_displacement=float(np.median(np.linalg.norm(delta, axis=1))),
        state_coverage=float(
            np.mean(np.std(np.asarray(dataset.observation, dtype=np.float64), axis=0))
        ),
        termination_rate=ended_terminated / dataset.episode_count,
        failure_rate=explicit_failure_count / dataset.episode_count,
        raw_transition_signal=float(np.median(np.linalg.norm(delta, axis=1))),
        prefix_costs=prefix_costs,
        collection_receipt=collection_receipt,
        semantic_style_hidden=bool(semantic_style_hidden),
    )


@dataclass(frozen=True)
class ProbeDistanceEvidence:
    task_id: str
    axis_id: str
    representation_id: str
    representation_protocol_digest: str
    semantic_bank_digests: Mapping[str, str]
    encoder_checkpoint_digest: str
    distance_matrix_digest: str
    independent_recompute_digest: str
    same_environment_cross_probe_distances: tuple[float, ...]
    different_dynamics_same_probe_distances: tuple[float, ...]
    repeated_bank_noise_distances: tuple[float, ...]
    probe_style_classifier_accuracy: float

    def __post_init__(self) -> None:
        for name in ("task_id", "axis_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ProbeAuditError(f"{name} must be a canonical non-empty string")
        if self.representation_id != R0_PADDED_RAW:
            raise ProbeAuditError(
                "probe distance evidence must use the frozen R0 padded-Raw representation"
            )
        _digest(
            self.representation_protocol_digest,
            "representation_protocol_digest",
        )
        if not isinstance(self.semantic_bank_digests, Mapping):
            raise ProbeAuditError("semantic_bank_digests must be a mapping")
        banks: dict[str, str] = {}
        for role, digest in self.semantic_bank_digests.items():
            if (
                not isinstance(role, str)
                or not role
                or role.strip() != role
            ):
                raise ProbeAuditError("semantic bank roles must be canonical strings")
            banks[role] = _digest(digest, f"semantic_bank_digests[{role!r}]")
        if len(banks) < 3:
            raise ProbeAuditError(
                "distance evidence must bind at least three semantic banks"
            )
        if len(set(banks.values())) < 2:
            raise ProbeAuditError("distance evidence semantic banks are collapsed")
        object.__setattr__(
            self,
            "semantic_bank_digests",
            MappingProxyType(dict(sorted(banks.items()))),
        )
        for name in (
            "encoder_checkpoint_digest",
            "distance_matrix_digest",
            "independent_recompute_digest",
        ):
            _digest(getattr(self, name), name)
        if self.distance_matrix_digest == self.independent_recompute_digest:
            raise ProbeAuditError(
                "independent recompute receipt must differ from distance matrix artifact"
            )
        for name in (
            "same_environment_cross_probe_distances",
            "different_dynamics_same_probe_distances",
            "repeated_bank_noise_distances",
        ):
            object.__setattr__(self, name, _distances(getattr(self, name), name))
        object.__setattr__(
            self,
            "probe_style_classifier_accuracy",
            _rate(
                self.probe_style_classifier_accuracy,
                "probe_style_classifier_accuracy",
            ),
        )

    @property
    def invariance_ratio(self) -> float:
        numerator = float(np.median(self.same_environment_cross_probe_distances))
        denominator = float(np.median(self.different_dynamics_same_probe_distances))
        return numerator / (denominator + np.finfo(np.float64).eps)

    @property
    def signal_to_noise_ratio(self) -> float:
        numerator = float(np.median(self.different_dynamics_same_probe_distances))
        denominator = float(np.median(self.repeated_bank_noise_distances))
        return numerator / (denominator + np.finfo(np.float64).eps)

    @property
    def median_dynamics_distance(self) -> float:
        return float(np.median(self.different_dynamics_same_probe_distances))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-distance-evidence.v2",
            "task_id": self.task_id,
            "axis_id": self.axis_id,
            "representation_id": self.representation_id,
            "representation_protocol_digest": self.representation_protocol_digest,
            "semantic_bank_digests": dict(self.semantic_bank_digests),
            "encoder_checkpoint_digest": self.encoder_checkpoint_digest,
            "distance_matrix_digest": self.distance_matrix_digest,
            "independent_recompute_digest": self.independent_recompute_digest,
            "same_environment_cross_probe_distances": list(
                self.same_environment_cross_probe_distances
            ),
            "different_dynamics_same_probe_distances": list(
                self.different_dynamics_same_probe_distances
            ),
            "repeated_bank_noise_distances": list(
                self.repeated_bank_noise_distances
            ),
            "probe_style_classifier_accuracy": self.probe_style_classifier_accuracy,
            "invariance_ratio": self.invariance_ratio,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
        }


@dataclass(frozen=True)
class ProbeGateThresholds:
    """Reviewable, digest-bound development thresholds for P2."""

    min_action_energy: float
    min_state_coverage: float
    min_raw_transition_signal: float
    min_different_dynamics_distance: float
    minimum_signal_to_noise_ratio: float
    maximum_invariance_ratio: float
    maximum_probe_style_classifier_accuracy: float
    max_saturation_rate: float
    max_termination_rate: float
    max_failure_rate: float

    def __post_init__(self) -> None:
        for name in (
            "min_action_energy",
            "min_state_coverage",
            "min_raw_transition_signal",
            "min_different_dynamics_distance",
            "minimum_signal_to_noise_ratio",
            "maximum_invariance_ratio",
        ):
            object.__setattr__(self, name, _finite_positive(getattr(self, name), name))
        for name in (
            "maximum_probe_style_classifier_accuracy",
            "max_saturation_rate",
            "max_termination_rate",
            "max_failure_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-gate-thresholds.v0",
            "min_action_energy": self.min_action_energy,
            "min_state_coverage": self.min_state_coverage,
            "min_raw_transition_signal": self.min_raw_transition_signal,
            "min_different_dynamics_distance": self.min_different_dynamics_distance,
            "minimum_signal_to_noise_ratio": self.minimum_signal_to_noise_ratio,
            "maximum_invariance_ratio": self.maximum_invariance_ratio,
            "maximum_probe_style_classifier_accuracy": self.maximum_probe_style_classifier_accuracy,
            "max_saturation_rate": self.max_saturation_rate,
            "max_termination_rate": self.max_termination_rate,
            "max_failure_rate": self.max_failure_rate,
        }


@dataclass(frozen=True)
class ProbeGateFreezeDecision:
    """A development freeze; formal G03-Probe authority is intentionally absent."""

    required_task_ids: tuple[str, ...]
    required_task_axis_pairs: tuple[tuple[str, str], ...]
    training_manifest_digest: str
    thresholds_digest: str
    decision_authority: str
    decision_status: Literal["DEVELOPMENT_FROZEN"] = "DEVELOPMENT_FROZEN"
    formal_authority_receipt_digest: None = None

    def __post_init__(self) -> None:
        tasks = tuple(self.required_task_ids)
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or any(
                not isinstance(task, str)
                or not task
                or task.strip() != task
                for task in tasks
            )
        ):
            raise ProbeAuditError(
                "freeze decision task IDs must be unique canonical strings"
            )
        object.__setattr__(self, "required_task_ids", tasks)
        pairs = tuple(tuple(pair) for pair in self.required_task_axis_pairs)
        if (
            not pairs
            or any(
                len(pair) != 2
                or any(
                    not isinstance(item, str)
                    or not item
                    or item.strip() != item
                    for item in pair
                )
                for pair in pairs
            )
            or len(set(pairs)) != len(pairs)
        ):
            raise ProbeAuditError(
                "required task/axis pairs must be unique canonical pairs"
            )
        if {task for task, _axis in pairs} != set(tasks):
            raise ProbeAuditError(
                "required task/axis pairs must cover exactly the frozen tasks"
            )
        if len({task for task, _axis in pairs}) < 2 or len(
            {_axis for _task, _axis in pairs}
        ) < 2:
            raise ProbeAuditError(
                "G03-Probe requires at least two tasks and two dynamics axes"
            )
        object.__setattr__(self, "required_task_axis_pairs", tuple(sorted(pairs)))
        _digest(self.training_manifest_digest, "training_manifest_digest")
        _digest(self.thresholds_digest, "thresholds_digest")
        if (
            not isinstance(self.decision_authority, str)
            or not self.decision_authority
            or self.decision_authority.strip() != self.decision_authority
        ):
            raise ProbeAuditError("decision_authority must be canonical and non-empty")
        if self.decision_status != "DEVELOPMENT_FROZEN":
            raise ProbeAuditError(
                "formal G03-Probe freeze authority is not implemented"
            )
        if self.formal_authority_receipt_digest is not None:
            raise ProbeAuditError(
                "formal G03-Probe authority receipt cannot be self-attested"
            )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-gate-freeze-decision.v0",
            "required_task_ids": list(self.required_task_ids),
            "required_task_axis_pairs": [list(pair) for pair in self.required_task_axis_pairs],
            "training_manifest_digest": self.training_manifest_digest,
            "thresholds_digest": self.thresholds_digest,
            "decision_authority": self.decision_authority,
            "decision_status": self.decision_status,
            "formal_authority_available": False,
            "formal_pass_eligible": False,
            "formal_authority_receipt_digest": None,
        }


ProbeGateStatus = Literal["DEVELOPMENT_PASS", "NO_GO_PROBE_COVERAGE"]


@dataclass(frozen=True)
class ProbeAuditReport:
    summaries: tuple[ProbeBankSummary, ...]
    distance_evidence: tuple[ProbeDistanceEvidence, ...]
    thresholds: ProbeGateThresholds
    freeze_decision: ProbeGateFreezeDecision
    training_manifest_digest: str
    gate_status: ProbeGateStatus
    failure_reasons: tuple[str, ...]
    cp2_holdout_pass: bool
    candidate_independence_pass: bool
    shared_target_banks_pass: bool

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, ProbeGateThresholds):
            raise ProbeAuditError("report thresholds must be typed")
        if not isinstance(self.freeze_decision, ProbeGateFreezeDecision):
            raise ProbeAuditError("report freeze decision must be typed")
        _digest(self.training_manifest_digest, "training_manifest_digest")
        if (
            self.freeze_decision.thresholds_digest != self.thresholds.digest
            or self.freeze_decision.training_manifest_digest
            != self.training_manifest_digest
        ):
            raise ProbeAuditError("report freeze bindings are inconsistent")
        if self.gate_status not in {"DEVELOPMENT_PASS", "NO_GO_PROBE_COVERAGE"}:
            raise ProbeAuditError(
                "formal PASS is unavailable until G03-Probe authority is implemented"
            )
        for name in (
            "cp2_holdout_pass",
            "candidate_independence_pass",
            "shared_target_banks_pass",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProbeAuditError(f"{name} must be boolean")
        reasons = tuple(str(reason) for reason in self.failure_reasons)
        if any(not reason or reason.strip() != reason for reason in reasons):
            raise ProbeAuditError("probe failure reasons must be canonical strings")
        if self.gate_status == "NO_GO_PROBE_COVERAGE" and not reasons:
            raise ProbeAuditError("probe No-Go requires failure reasons")
        if self.gate_status != "NO_GO_PROBE_COVERAGE" and reasons:
            raise ProbeAuditError("passing probe audit cannot carry failure reasons")
        core_flags = (
            self.cp2_holdout_pass,
            self.candidate_independence_pass,
            self.shared_target_banks_pass,
        )
        if self.gate_status == "DEVELOPMENT_PASS" and not all(core_flags):
            raise ProbeAuditError(
                "development PASS is inconsistent with failed audit flags"
            )
        flag_reasons = (
            (self.cp2_holdout_pass, "CP2_HOLDOUT_VIOLATION"),
            (
                self.candidate_independence_pass,
                "CANDIDATE_INDEPENDENCE_VIOLATION",
            ),
            (self.shared_target_banks_pass, "TARGET_PROBE_BANK_MISMATCH"),
        )
        for flag, reason in flag_reasons:
            if flag == (reason in reasons):
                raise ProbeAuditError(
                    f"audit flag is inconsistent with failure reason {reason}"
                )
        object.__setattr__(self, "failure_reasons", reasons)

    @property
    def evidence_scope(self) -> str:
        return "DEVELOPMENT"

    @property
    def formal_authority_available(self) -> bool:
        return False

    @property
    def formal_pass_eligible(self) -> bool:
        return False

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-probe-audit-report.v1",
            "protocol_id": PROBE_AUDIT_PROTOCOL_ID,
            "summaries": [summary.to_dict() for summary in self.summaries],
            "distance_evidence": [evidence.to_dict() for evidence in self.distance_evidence],
            "thresholds": self.thresholds.to_dict(),
            "thresholds_digest": self.thresholds.digest,
            "freeze_decision": self.freeze_decision.to_dict(),
            "freeze_decision_digest": self.freeze_decision.digest,
            "training_manifest_digest": self.training_manifest_digest,
            "evidence_scope": self.evidence_scope,
            "gate_status": self.gate_status,
            "failure_reasons": list(self.failure_reasons),
            "cp2_holdout_pass": self.cp2_holdout_pass,
            "candidate_independence_pass": self.candidate_independence_pass,
            "shared_target_banks_pass": self.shared_target_banks_pass,
            "formal_authority_available": self.formal_authority_available,
            "formal_pass_eligible": self.formal_pass_eligible,
        }


def _shared_target_banks(
    bindings: Mapping[str, Mapping[str, str]],
) -> bool:
    if not bindings:
        return False
    normalized = []
    for index in bindings.values():
        rows: list[tuple[str, str]] = []
        for context, digest in index.items():
            if not isinstance(context, str) or not context:
                return False
            if not isinstance(digest, str) or len(digest) != 64:
                return False
            try:
                int(digest, 16)
            except ValueError:
                return False
            rows.append((context, digest))
        normalized.append(tuple(sorted(rows)))
    return bool(normalized[0]) and all(value == normalized[0] for value in normalized[1:])


def evaluate_probe_gate(
    *,
    summaries: Sequence[ProbeBankSummary],
    distance_evidence: Sequence[ProbeDistanceEvidence],
    training_manifest: ProbeTrainingManifest,
    thresholds: ProbeGateThresholds,
    freeze_decision: ProbeGateFreezeDecision,
    target_bank_bindings_by_encoder: Mapping[str, Mapping[str, str]],
) -> ProbeAuditReport:
    """Evaluate the P2 development gate; formal PASS cannot be emitted here."""

    summary_tuple = tuple(summaries)
    evidence_tuple = tuple(distance_evidence)
    if not all(isinstance(row, ProbeBankSummary) for row in summary_tuple):
        raise ProbeAuditError("summaries must contain typed ProbeBankSummary rows")
    if not all(isinstance(row, ProbeDistanceEvidence) for row in evidence_tuple):
        raise ProbeAuditError(
            "distance_evidence must contain typed ProbeDistanceEvidence rows"
        )
    if not isinstance(training_manifest, ProbeTrainingManifest):
        raise ProbeAuditError("training_manifest must be typed")
    if not isinstance(thresholds, ProbeGateThresholds):
        raise ProbeAuditError("thresholds must be a typed frozen record")
    if not isinstance(freeze_decision, ProbeGateFreezeDecision):
        raise ProbeAuditError("freeze_decision must be a typed development record")
    if freeze_decision.training_manifest_digest != training_manifest.digest:
        raise ProbeAuditError("freeze decision binds another training manifest")
    if freeze_decision.thresholds_digest != thresholds.digest:
        raise ProbeAuditError("freeze decision binds another threshold record")
    tasks = freeze_decision.required_task_ids
    try:
        validate_cp2_holdout(training_manifest)
        cp2_holdout_pass = True
    except ProbeContractError:
        cp2_holdout_pass = False
    candidate_pass = all(record.candidate_independence_pass for record in summary_tuple)
    shared_banks_pass = _shared_target_banks(target_bank_bindings_by_encoder)
    reasons: list[str] = []
    if not cp2_holdout_pass:
        reasons.append("CP2_HOLDOUT_VIOLATION")
    if not candidate_pass:
        reasons.append("CANDIDATE_INDEPENDENCE_VIOLATION")
    if not shared_banks_pass:
        reasons.append("TARGET_PROBE_BANK_MISMATCH")
    for task_id in tasks:
        task_rows = [record for record in summary_tuple if record.task_id == task_id]
        regimes = {record.regime for record in task_rows}
        if not {CP0_EXACT_COMMON, CP2_UNSEEN_PROBE}.issubset(regimes):
            reasons.append(f"MISSING_CP0_OR_CP2:{task_id}")
        for record in task_rows:
            if not record.finite:
                reasons.append(f"NONFINITE:{task_id}:{record.probe_style_id}")
            if record.action_energy < thresholds.min_action_energy:
                reasons.append(f"LOW_ACTION_ENERGY:{task_id}:{record.probe_style_id}")
            if record.state_coverage < thresholds.min_state_coverage:
                reasons.append(f"LOW_STATE_COVERAGE:{task_id}:{record.probe_style_id}")
            if record.raw_transition_signal < thresholds.min_raw_transition_signal:
                reasons.append(
                    f"NONPOSITIVE_RAW_DYNAMICS:{task_id}:{record.probe_style_id}"
                )
            if record.action_saturation_rate > thresholds.max_saturation_rate:
                reasons.append(f"ACTION_SATURATION:{task_id}:{record.probe_style_id}")
            if (
                record.termination_rate > thresholds.max_termination_rate
                or record.failure_rate > thresholds.max_failure_rate
            ):
                reasons.append(f"SAFETY_FAILURE:{task_id}:{record.probe_style_id}")
            if not record.semantic_style_hidden:
                reasons.append(f"STYLE_LABEL_LEAKAGE:{task_id}:{record.probe_style_id}")
        task_axes = tuple(
            axis_id
            for registered_task, axis_id in freeze_decision.required_task_axis_pairs
            if registered_task == task_id
        )
        for axis_id in task_axes:
            task_evidence = [
                row
                for row in evidence_tuple
                if row.task_id == task_id and row.axis_id == axis_id
            ]
            if len(task_evidence) != 1:
                reasons.append(f"MISSING_DYNAMICS_EVIDENCE:{task_id}:{axis_id}")
                continue
            evidence = task_evidence[0]
            if (
                evidence.median_dynamics_distance
                < thresholds.min_different_dynamics_distance
            ):
                reasons.append(f"SEMANTIC_BANK_COLLAPSE:{task_id}:{axis_id}")
            if (
                evidence.signal_to_noise_ratio
                < thresholds.minimum_signal_to_noise_ratio
            ):
                reasons.append(
                    f"RAW_DYNAMICS_BELOW_BANK_NOISE:{task_id}:{axis_id}"
                )
            if evidence.invariance_ratio > thresholds.maximum_invariance_ratio:
                reasons.append(f"PROBE_INVARIANCE_FAILURE:{task_id}:{axis_id}")
            if (
                evidence.probe_style_classifier_accuracy
                > thresholds.maximum_probe_style_classifier_accuracy
            ):
                reasons.append(
                    f"PROBE_STYLE_CLASSIFIER_TOO_ACCURATE:{task_id}:{axis_id}"
                )
            shared_task_digest = next(
                iter(target_bank_bindings_by_encoder.values()), {}
            ).get(task_id)
            if shared_task_digest not in set(evidence.semantic_bank_digests.values()):
                reasons.append(
                    f"DISTANCE_EVIDENCE_MISSING_TARGET_BANK:{task_id}:{axis_id}"
                )
    unknown_summaries = {record.task_id for record in summary_tuple} - set(tasks)
    if unknown_summaries:
        reasons.append("UNREGISTERED_TASK_EVIDENCE:" + ",".join(sorted(unknown_summaries)))
    registered_pairs = set(freeze_decision.required_task_axis_pairs)
    unknown_distance_pairs = {
        (record.task_id, record.axis_id) for record in evidence_tuple
    } - registered_pairs
    if unknown_distance_pairs:
        reasons.append(
            "UNREGISTERED_DISTANCE_EVIDENCE:"
            + ",".join(
                f"{task_id}/{axis_id}"
                for task_id, axis_id in sorted(unknown_distance_pairs)
            )
        )
    unique_reasons = tuple(dict.fromkeys(reasons))
    if unique_reasons:
        status: ProbeGateStatus = "NO_GO_PROBE_COVERAGE"
    else:
        status = "DEVELOPMENT_PASS"
    return ProbeAuditReport(
        summaries=summary_tuple,
        distance_evidence=evidence_tuple,
        thresholds=thresholds,
        freeze_decision=freeze_decision,
        training_manifest_digest=training_manifest.digest,
        gate_status=status,
        failure_reasons=unique_reasons,
        cp2_holdout_pass=cp2_holdout_pass,
        candidate_independence_pass=candidate_pass,
        shared_target_banks_pass=shared_banks_pass,
    )
