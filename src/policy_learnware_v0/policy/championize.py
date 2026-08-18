"""Pure, deterministic source-side policy championization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .bundle import PolicyBundleMetadata


@dataclass(frozen=True)
class CandidateEvaluation:
    """Private source-only evaluation; this never enters a selector entry."""

    task: str
    algorithm: str
    training_seed: int
    outer_iteration: int
    environment_steps: int
    bundle_dir: Path
    bundle_digest: str
    episode_returns: tuple[float, ...]
    checksum_passed: bool = True
    parity_passed: bool = True

    @classmethod
    def from_bundle(
        cls,
        metadata: PolicyBundleMetadata,
        episode_returns: Sequence[float],
        *,
        parity_passed: bool,
    ) -> "CandidateEvaluation":
        return cls(
            task=metadata.task,
            algorithm=metadata.algorithm,
            training_seed=metadata.training_seed,
            outer_iteration=metadata.outer_iteration,
            environment_steps=metadata.environment_steps,
            bundle_dir=metadata.bundle_dir,
            bundle_digest=metadata.bundle_digest,
            episode_returns=tuple(float(value) for value in episode_returns),
            checksum_passed=True,
            parity_passed=bool(parity_passed),
        )


@dataclass(frozen=True)
class RankedCandidate:
    candidate: CandidateEvaluation
    mean_return: float
    return_std: float


@dataclass(frozen=True)
class CandidateRejection:
    task: str
    bundle_digest: str
    reason: str


@dataclass(frozen=True)
class TaskChampion:
    task: str
    selected: RankedCandidate
    ranking: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class ChampionizationResult:
    champions: tuple[TaskChampion, ...]
    rejected: tuple[CandidateRejection, ...]
    selection_rule: str = "max_mean_then_min_std_then_bundle_digest"

    @property
    def by_task(self) -> Mapping[str, TaskChampion]:
        return MappingProxyType({champion.task: champion for champion in self.champions})


def _rank(candidate: CandidateEvaluation) -> RankedCandidate:
    returns = tuple(float(value) for value in candidate.episode_returns)
    if not returns:
        raise ValueError("episode_returns is empty")
    if any(not math.isfinite(value) for value in returns):
        raise ValueError("episode_returns contains a non-finite value")
    mean = fmean(returns)
    std = pstdev(returns)
    return RankedCandidate(candidate=candidate, mean_return=mean, return_std=std)


def championize(
    candidates: Iterable[CandidateEvaluation],
    *,
    checkpoint_outer: int,
    expected_environment_steps: int | None = None,
    expected_candidates_per_task: int | None = None,
    expected_tasks: Iterable[str] | None = None,
) -> ChampionizationResult:
    """Choose exactly one validated fixed-budget candidate per source task.

    Sorting implements the pre-registered rule exactly: higher mean, lower
    population standard deviation, then lexicographically smaller bundle
    digest.  No selector output or target return is accepted by this API.
    """

    grouped: dict[str, list[RankedCandidate]] = {}
    rejected: list[CandidateRejection] = []
    seen_digests: set[str] = set()
    for candidate in candidates:
        reason: str | None = None
        if candidate.bundle_digest in seen_digests:
            reason = "duplicate bundle digest"
        elif not candidate.checksum_passed:
            reason = "checksum validation failed"
        elif not candidate.parity_passed:
            reason = "golden parity failed"
        elif candidate.outer_iteration != checkpoint_outer:
            reason = f"outer {candidate.outer_iteration} != fixed outer {checkpoint_outer}"
        elif (
            expected_environment_steps is not None
            and candidate.environment_steps != expected_environment_steps
        ):
            reason = "environment-step budget mismatch"
        try:
            ranked = _rank(candidate) if reason is None else None
        except ValueError as error:
            reason = str(error)
            ranked = None
        seen_digests.add(candidate.bundle_digest)
        if reason is not None:
            rejected.append(CandidateRejection(candidate.task, candidate.bundle_digest, reason))
            continue
        assert ranked is not None
        grouped.setdefault(candidate.task, []).append(ranked)

    expected = set(expected_tasks) if expected_tasks is not None else set(grouped)
    missing = sorted(expected.difference(grouped))
    unexpected = sorted(set(grouped).difference(expected))
    if missing or unexpected:
        raise ValueError(f"task coverage mismatch: missing={missing}, unexpected={unexpected}")

    champions: list[TaskChampion] = []
    for task in sorted(expected):
        ranked_candidates = grouped[task]
        if expected_candidates_per_task is not None and len(ranked_candidates) != int(
            expected_candidates_per_task
        ):
            raise ValueError(
                f"task {task!r} has {len(ranked_candidates)} valid candidates, "
                f"expected {expected_candidates_per_task}"
            )
        ranking = tuple(
            sorted(
                ranked_candidates,
                key=lambda item: (
                    -item.mean_return,
                    item.return_std,
                    item.candidate.bundle_digest,
                ),
            )
        )
        if not ranking:
            raise ValueError(f"task {task!r} has no valid candidate")
        champions.append(TaskChampion(task=task, selected=ranking[0], ranking=ranking))
    return ChampionizationResult(tuple(champions), tuple(rejected))
